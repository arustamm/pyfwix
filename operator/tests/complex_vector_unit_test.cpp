#include <complex4DReg.h>
#include <gtest/gtest.h>
#include <complex_vector.h>
#include <cuda_runtime.h>
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



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}