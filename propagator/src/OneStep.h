#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <RefSampler.h>
#include <PhaseShift.h>
#include <Selector.h>
#include <FFT.h>
#include <Taper.h>
  // operator to propagate 2D wavefield ONCE in (x-y) for multiple sources and freqs (ns-nw) 
class OneStep : public CudaOperator<complex4DReg, complex4DReg>  {
public:
  OneStep (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par, 
  complex_vector* model = nullptr, complex_vector* data = nullptr, 
  dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream) {

    initialize(domain, slow->getHyper(), par);
    _ref_ = std::make_shared<RefSampler>(slow, par);
  };

  OneStep (const std::shared_ptr<hypercube>& domain, std::shared_ptr<hypercube> slow_hyper, std::shared_ptr<paramObj> par, 
    std::shared_ptr<RefSampler> ref = nullptr,
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
    CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream) {
  
      initialize(domain, slow_hyper, par);
      if (ref == nullptr)
        _ref_ = std::make_shared<RefSampler>(slow_hyper, par);
      else
        _ref_ = ref;
    };

  virtual ~OneStep() {
    _wfld_ref->~complex_vector();
    CHECK_CUDA_ERROR(cudaFree(_wfld_ref));
    model_k->~complex_vector();
    CHECK_CUDA_ERROR(cudaFree(model_k));
  };

  inline void set_depth(size_t iz) {
    _iz_ = iz;
    select->set_labels(_ref_->get_ref_labels(iz));
  };
  size_t get_depth() {return _iz_;};

  complex_vector* get_ref_wfld() {return _wfld_ref;};

protected:
  complex_vector* _wfld_ref;
  complex_vector* model_k;
  size_t _nref_, _iz_;
  float _dz_;
  std::shared_ptr<RefSampler> _ref_;
  std::unique_ptr<PhaseShift> ps;
  std::unique_ptr<cuFFT2d> fft2d;
  std::unique_ptr<Selector> select;
  std::unique_ptr<Taper> taper;

private:

  void initialize(std::shared_ptr<hypercube> domain, std::shared_ptr<hypercube> slow_hyper, std::shared_ptr<paramObj> par) {
    _nref_ = par->getInt("nref",1);
    taper = std::make_unique<Taper>(domain, par, model_vec, data_vec, _grid_, _block_, _stream_);
    ps = std::make_unique<PhaseShift>(domain, slow_hyper->getAxis(4).d, par->getFloat("eps",0.), model_vec, data_vec, _grid_, _block_, _stream_);
    _wfld_ref = model_vec->cloneSpace();
    model_k = model_vec->cloneSpace();
  
    fft2d = std::make_unique<cuFFT2d>(domain, model_vec, data_vec, _grid_, _block_, _stream_);
    select = std::make_unique<Selector>(domain, model_vec, data_vec, _grid_, _block_, _stream_);
  }

};

class PSPI : public OneStep {
public:
  PSPI (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par,
  complex_vector* model = nullptr, complex_vector* data = nullptr, dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  OneStep(domain, slow, par, model, data, grid, block, stream) {};

  PSPI (const std::shared_ptr<hypercube>& domain, std::shared_ptr<hypercube> slow_hyper, std::shared_ptr<paramObj> par,
    std::shared_ptr<RefSampler> ref = nullptr,
    complex_vector* model = nullptr, complex_vector* data = nullptr, dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
    OneStep(domain, slow_hyper, par, ref, model, data, grid, block, stream) {};

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_forward (complex_vector* __restrict__ model);

  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (complex_vector* __restrict__ data);

  // void cu_inverse (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
};

class NSPS : public OneStep {
public:
  NSPS (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par,
  complex_vector* model = nullptr, complex_vector* data = nullptr, dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  OneStep(domain, slow, par, model, data, grid, block, stream) {};

  NSPS (const std::shared_ptr<hypercube>& domain, std::shared_ptr<hypercube> slow_hyper, 
    std::shared_ptr<paramObj> par,
    std::shared_ptr<RefSampler> ref = nullptr,
    complex_vector* model = nullptr, complex_vector* data = nullptr, dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
    OneStep(domain, slow_hyper, par, ref, model, data, grid, block, stream) {};

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  // void cu_inverse (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
};