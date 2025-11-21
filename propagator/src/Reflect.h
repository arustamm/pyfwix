#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <OneStep.h>
#include <prop_kernels.cuh>

// reflecting wavefields in the volume [nz, ns, nw, ny, nx] 
class BaseReflect : public CudaOperator<complex4DReg, complex4DReg>  {
public:
  BaseReflect (const std::shared_ptr<hypercube>& domain, 
    const std::vector<std::shared_ptr<complex4DReg>>& slow_impedance, 
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0);

  BaseReflect (const std::shared_ptr<hypercube>& domain, std::shared_ptr<hypercube> slow_hyper, 
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0);

  void set_depth(int iz) {
    if (!isModelSet) 
      throw std::runtime_error("Model not set in Reflection operator");

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

  virtual ~BaseReflect() { 
    d_slow_slice->~complex_vector();
    CHECK_CUDA_ERROR(cudaFree(d_slow_slice));
    d_den_slice->~complex_vector();
    CHECK_CUDA_ERROR(cudaFree(d_den_slice));
   };

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
    throw std::runtime_error("Forward not implemented for BaseReflect. Use derived classes.");
  };
  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
    throw std::runtime_error("Adjoint not implemented for BaseReflect. Use derived classes.");
  }

  void cu_forward (complex_vector* __restrict__ model) {
    throw std::runtime_error("Forward not implemented for BaseReflect. Use derived classes.");
  }
  void cu_adjoint (complex_vector* __restrict__ data) {
    throw std::runtime_error("Adjoint not implemented for BaseReflect. Use derived classes.");
  }

  void set_background_model(const std::vector<std::shared_ptr<complex4DReg>>& slow_impedance) {
    _slow = slow_impedance[0];
    _density = slow_impedance[1];
    isModelSet = true;
  }

protected:

  void initialize(std::shared_ptr<hypercube> slow_hyper);

  std::shared_ptr<complex4DReg> _slow, _density;
  int nw, ny, nx, ns, nz;
  size_t slice_size;
  complex_vector *d_slow_slice, *d_den_slice;
  bool isModelSet = false;
};

// reflecting wavefields in the volume [nz, ns, nw, ny, nx] 
class Reflect : public BaseReflect  {
public:
  Reflect (const std::shared_ptr<hypercube>& domain, 
    const std::vector<std::shared_ptr<complex4DReg>>& slow_impedance, 
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  BaseReflect(domain, slow_impedance, model, data, grid, block, stream) {
    launcher = Refl_launcher(&refl_forward, &refl_adjoint, _grid_, _block_, _stream_);
    launcher_in_place = Refl_launcher(&refl_forward_in, &refl_adjoint_in, _grid_, _block_, _stream_);
  };

  Reflect (const std::shared_ptr<hypercube>& domain, std::shared_ptr<hypercube> slow_hyper, 
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  BaseReflect(domain, slow_hyper, model, data, grid, block, stream) {
    launcher = Refl_launcher(&refl_forward, &refl_adjoint, _grid_, _block_, _stream_);
    launcher_in_place = Refl_launcher(&refl_forward_in, &refl_adjoint_in, _grid_, _block_, _stream_);
  }

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);

  void cu_forward (complex_vector* __restrict__ model);
  void cu_adjoint (complex_vector* __restrict__ data);

private:
  Refl_launcher launcher, launcher_in_place;

};