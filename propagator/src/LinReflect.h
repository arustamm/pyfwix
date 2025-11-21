#pragma once
#include <CudaOperator.h>
#include <complex5DReg.h>
#include <complex4DReg.h>
#include <complex3DReg.h>
#include <prop_kernels.cuh>

class LinReflect : public CudaOperator<complex5DReg, complex3DReg> {
public:
  LinReflect (
    const std::shared_ptr<hypercube>& domain,
    const std::shared_ptr<hypercube>& range, 
    const std::vector<std::shared_ptr<complex4DReg>>& slow_impedance, 
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0);

  virtual ~LinReflect() { 
    d_slow_slice->~complex_vector();
    CHECK_CUDA_ERROR(cudaFree(d_slow_slice));
    d_den_slice->~complex_vector();
    CHECK_CUDA_ERROR(cudaFree(d_den_slice));
   };

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  
  void set_depth(int iz) {
    // copy 2 slices of the model
    size_t offset = iz * slice_size;
    if (iz < nz-1) {
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_slow_slice->mat, _slow->getVals() + offset, 2*slice_size*sizeof(std::complex<float>), cudaMemcpyHostToDevice, _stream_));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_den_slice->mat, _density->getVals() + offset, 2*slice_size*sizeof(std::complex<float>), cudaMemcpyHostToDevice, _stream_));
    }
    else {
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_slow_slice->mat, _slow->getVals() + offset, slice_size*sizeof(std::complex<float>), cudaMemcpyHostToDevice, _stream_));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_den_slice->mat, _density->getVals() + offset, slice_size*sizeof(std::complex<float>), cudaMemcpyHostToDevice, _stream_));
      // add the same layer to effectively have 0 reflection coeffecient
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_slow_slice->mat + slice_size, d_slow_slice->mat, slice_size*sizeof(std::complex<float>), cudaMemcpyDeviceToDevice, _stream_));
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_den_slice->mat + slice_size, d_den_slice->mat, slice_size*sizeof(std::complex<float>), cudaMemcpyDeviceToDevice, _stream_));
    }
  }

  void set_background_model(const std::vector<std::shared_ptr<complex4DReg>>& slow_impedance) {
    _slow = slow_impedance[0];
    _density = slow_impedance[1];
  }
    

private:
    Refl_launcher launcher;
    size_t slice_size;
    complex_vector *d_slow_slice, *d_den_slice;
    std::shared_ptr<complex4DReg> _slow, _density;
    int nw, ny, nx, ns, nz;

};