#include <complex4DReg.h>
#include <gtest/gtest.h>
#include <complex_vector.h>
#include <cuda_runtime.h>
// #include "SZ3/api/sz.hpp"
#include "zfp.h"

class ComplexVectorTest : public testing::Test {
 protected:
  void SetUp() override {
    n1 = 101;
    n2 = 100;
    n3 = 20;
    n4 = 10;
    hyper = std::make_shared<hypercube>(n1, n2, n3, n4);
    vec = make_complex_vector(hyper, {2,2,2}, {2,2,2});

  }

  void TearDown() override {
    vec->~complex_vector();
    CHECK_CUDA_ERROR(cudaFree(vec));
  }

  std::shared_ptr<hypercube> hyper;
  complex_vector* vec;
  int n1, n2, n3, n4;
};

TEST_F(ComplexVectorTest, check_flat_index) { 
  int ind1[] = {0,0,0,1};
  int ind2[] = {0,0,1,1};
  int ind3[] = {0,1,0,0};
  int dims[] = {n4, n3, n2, n1};
  ASSERT_EQ(ND_TO_FLAT(ind1, dims), 1);
  ASSERT_EQ(ND_TO_FLAT(ind2, dims), 102);
  ASSERT_EQ(ND_TO_FLAT(ind3, dims), 10100);
}

TEST_F(ComplexVectorTest, check_complex_vector) { 
  int ndim = vec->ndim;

  int *n = new int[ndim];
  float *d = new float[ndim];
  float *o = new float[ndim];
  cudaMemcpy(n, vec->n, sizeof(int)*ndim, cudaMemcpyDeviceToHost);
  cudaMemcpy(d, vec->d, sizeof(float)*ndim, cudaMemcpyDeviceToHost);
  cudaMemcpy(o, vec->o, sizeof(float)*ndim, cudaMemcpyDeviceToHost);
  for (int i=0; i < ndim; ++i) {
    ASSERT_EQ(n[i], hyper->getAxis(i+1).n);
    ASSERT_EQ(d[i], hyper->getAxis(i+1).d);
    ASSERT_EQ(o[i], hyper->getAxis(i+1).o);
  }
  // vec->compress();
  delete[] n;
  delete[] o;
  delete[] d;
}


TEST_F(ComplexVectorTest, make_view) {
  // Create a view using make_view
  int start = 4;
  int end = 9;
  complex_vector* view = vec->make_view(start,end);

  // Check dimensions and size
  ASSERT_EQ(view->ndim, vec->ndim);
  ASSERT_EQ(view->nelem, vec->nelem * (end-start) / n4);

  // Check metadata
  for (int i = 0; i < view->ndim-1; ++i) {
    ASSERT_EQ(view->n[i], vec->n[i]);
    ASSERT_EQ(view->d[i], vec->d[i]);
    ASSERT_EQ(view->o[i], vec->o[i]);
  }

  ASSERT_EQ(view->o[n4-1], vec->o[n4-1] + start*vec->d[n4-1]);

  // Check that the view is not allocated
  ASSERT_FALSE(view->allocated);

  view->~complex_vector();
  CHECK_CUDA_ERROR(cudaFree(view));
}

TEST_F(ComplexVectorTest, view_modify) {
  // fill in with const values
  auto cpu_vec = std::make_shared<complex4DReg>(hyper);
  cpu_vec->set(1.f);
  // copy to gpu_vec
  CHECK_CUDA_ERROR(cudaMemcpy(vec->mat, cpu_vec->getVals(), hyper->getN123() * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
  // create a view
  // actually take a view
  int index = 5;
  complex_vector* view = vec->make_view(index, index+1);
  // zero out the slice
  view->zero();
  // copy back 
  CHECK_CUDA_ERROR(cudaMemcpy(cpu_vec->getVals(), vec->mat, hyper->getN123() * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
  // check that it modified the original vector
  for (int i4 = 0; i4 < n4; ++i4) {
    for (int i3 = 0; i3 < n3; ++i3) {
      for (int i2 = 0; i2 < n2; ++i2) {
        for (int i1 = 0; i1 < n1; ++i1) {
          auto val = (*cpu_vec->_mat)[i4][i3][i2][i1];
          if (i4 == index) ASSERT_EQ(val, 0.f);
          else ASSERT_EQ(val, 1.f);
        }
      }
    }   
  }

  view->~complex_vector();
  CHECK_CUDA_ERROR(cudaFree(view));

}

TEST_F(ComplexVectorTest, slice_modify) {
  // fill in with const values
  auto cpu_vec = std::make_shared<complex4DReg>(hyper);
  cpu_vec->set(1.f);
  // copy to gpu_vec
  CHECK_CUDA_ERROR(cudaMemcpy(vec->mat, cpu_vec->getVals(), hyper->getN123() * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
  // create a slice
  auto slice = vec->make_slice();
  // actually slice
  int index = 5;
  vec->slice_at(slice, index);
  // zero out the slice
  slice->zero();
  // copy back 
  CHECK_CUDA_ERROR(cudaMemcpy(cpu_vec->getVals(), vec->mat, hyper->getN123() * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
  // check that it modified the original vector
  for (int i4 = 0; i4 < n4; ++i4) {
    for (int i3 = 0; i3 < n3; ++i3) {
      for (int i2 = 0; i2 < n2; ++i2) {
        for (int i1 = 0; i1 < n1; ++i1) {
          auto val = (*cpu_vec->_mat)[i4][i3][i2][i1];
          if (i4 == index) ASSERT_EQ(val, 0.f);
          else ASSERT_EQ(val, 1.f);
        }
      }
    }   
  }

  slice->~complex_vector();
  CHECK_CUDA_ERROR(cudaFree(slice));

}

TEST_F(ComplexVectorTest, compress_decompress) {
  auto orig = std::make_shared<complex4DReg>(hyper);
  auto decomp = std::make_shared<complex4DReg>(hyper);

// Loop through each dimension to populate 'orig' with plane wave data
  // This section generates a wavelike signal in the 'orig' data.
  float dx = 0.01f;
  float dy = 0.01f;
  float f_min_hz = 1.0f;  // Minimum frequency in Hz
  float f_max_hz = 20.0f; // Maximum frequency in Hz

  for (int i4 = 0; i4 < n4; ++i4) {
    float sx = i4 * dx;
    float sy = i4 * dy;
    for (int i3 = 0; i3 < n3; ++i3) {
      // Calculate frequency in Hz, ranging from f_min_hz to f_max_hz
      float frequency_hz = f_min_hz + i3 * (f_max_hz - f_min_hz) / (n3 - 1.0f);
      // Convert to angular frequency
      float w = frequency_hz * 2.0f * M_PI;
      for (int i2 = 0; i2 < n2; ++i2) {
        for (int i1 = 0; i1 < n1; ++i1) {
          // Calculate the wave vector components
          float kx = 2.0f * M_PI * w / dx;
          float ky = 2.0f * M_PI * w / dy;
          // Calculate the phase
          float phase = kx * (i1 * dx - sx) + ky * (i2 * dy - sy);
          // Generate the plane wave value
          (*orig->_mat)[i4][i3][i2][i1] = std::polar(1.0f, phase); // Amplitude 1, phase calculated above
        }
      }
    }
  }

  // Get a float pointer to the original complex data (interleaved real/imaginary parts)
  float* data_to_compress = reinterpret_cast<float*>(orig->getVals());

  // --- ZFP Compression Setup ---
  zfp_stream* zfp_stream = zfp_stream_open(NULL); // Allocate zfp stream
  zfp_field* zfp_field = zfp_field_4d(data_to_compress, zfp_type_float, 2*n1, n2, n3, n4);

  // Set compression parameters using accuracy mode (equivalent to SZ3's relative error bound)
  double relErrorBound = 1E-3;
  zfp_stream_set_accuracy(zfp_stream, relErrorBound); // This sets the maximum error tolerance

  // Estimate maximum size for compressed data
  size_t max_compressed_bufsize = zfp_stream_maximum_size(zfp_stream, zfp_field);
  char *compressedData = new char[max_compressed_bufsize]; // Allocate buffer for compressed data

  // Open a bitstream and associate it with the compressed data buffer
  bitstream* stream = stream_open(compressedData, max_compressed_bufsize);
  zfp_stream_set_bit_stream(zfp_stream, stream);
  zfp_stream_rewind(zfp_stream);

  // Perform compression
  size_t actual_compressed_size = zfp_compress(zfp_stream, zfp_field);
  if (actual_compressed_size == 0) {
      std::cerr << "ZFP compression failed in test!" << std::endl;
      ASSERT_NE(actual_compressed_size, 0) << "ZFP compression failed!";
      delete[] compressedData;
      stream_close(stream);
      zfp_stream_close(zfp_stream);
      zfp_field_free(zfp_field);
      return;
  }
  stream_close(stream); // Close the bitstream (does not free compressedData buffer)

  // --- ZFP Decompression ---
  float* outdata = reinterpret_cast<float*>(decomp->getVals()); // Get pointer for decompressed output
  zfp_field_set_pointer(zfp_field, (void*)outdata);             // Set output data pointer for field

  // Reopen bitstream with the *compressed* data for decompression
  stream = stream_open(compressedData, actual_compressed_size);
  zfp_stream_set_bit_stream(zfp_stream, stream);
  zfp_stream_rewind(zfp_stream);

  // Perform decompression
  int success = zfp_decompress(zfp_stream, zfp_field);
  if (!success) {
      std::cerr << "ZFP decompression failed in test!" << std::endl;
      ASSERT_TRUE(false) << "ZFP decompression failed!";
  }
  stream_close(stream);

  // --- Cleanup ZFP resources ---
  zfp_stream_close(zfp_stream);
  zfp_field_free(zfp_field);

  // --- Error and Compression Ratio Calculation ---
  decomp->scaleAdd(orig, 1, -1); // decomp = decomp - orig
  // Calculate relative L2 error
  double err = sqrt(std::abs(decomp->dot(decomp)) / std::abs(orig->dot(orig)));
  std::cout << "ZFP Reconstruction Error (Relative L2): " << err << "\n";

  // Original data size (2 floats per complex<float>)
  size_t originalSize = (size_t)n4 * n3 * n2 * n1 * sizeof(std::complex<float>);
  // Calculate and print the compression ratio
  double compressionRatio = static_cast<double>(originalSize) / actual_compressed_size;
  std::cout << "ZFP Compression Ratio: " << compressionRatio << std::endl;

  delete[] compressedData; // Free the allocated compressed data buffer
}

//   const float* data = reinterpret_cast<const float*>(orig->getVals());

//   SZ3::Config conf(n4,n3,n2,2*n1);
//   conf.cmprAlgo = SZ3::ALGO_INTERP_LORENZO;
//   conf.errorBoundMode = SZ3::EB_REL; // refer to def.hpp for all supported error bound mode
//   conf.relErrorBound = 1E-3; // absolute error bound 1e-3
//   size_t outSize;
//   char *compressedData = SZ_compress(conf, data, outSize);

//   // decompress
//   float* outdata = reinterpret_cast<float*>(decomp->getVals());
//   SZ_decompress(conf, compressedData, outSize, outdata);

//   decomp->scaleAdd(orig, 1, -1);
//   double err = sqrt(std::real(decomp->dot(decomp)) / std::real(orig->dot(orig)));
//   std::cout << "Error: " << err << "\n";

//   size_t originalSize = 2 * n4 * n3 * n2 * n1 * sizeof(float); 
//   // Calculate and print the compression ratio
//   double compressionRatio = static_cast<double>(originalSize) / outSize;
//   std::cout << "Compression ratio: " << compressionRatio << std::endl;

//   delete[] compressedData;
// }




int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}