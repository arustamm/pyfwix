#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <OneStep.h>
#include <Reflect.h>

#include <sep_reg_file.h>
#include <utils.h>
#include <ioModes.h>
#include <WavefieldPool.h>

// propagating wavefields in the volume [nz, ns, nw, ny, nx] from 0 to nz-1
class OneWay : public CudaOperator<complex4DReg, complex4DReg>  {
public:
  OneWay (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par, 
  std::shared_ptr<WavefieldPool> wfld_pool = nullptr,
  complex_vector* model = nullptr, complex_vector* data = nullptr, 
  dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream),
  _slow_(slow), _param(par) {

    initialize(domain, slow->getHyper(), par, wfld_pool);
    // for now only support PSPI propagator
    prop = std::make_shared<PSPI>(domain, slow, par, model_vec, data_vec, _grid_, _block_, _stream_);

  };

  OneWay (const std::shared_ptr<hypercube>& domain, std::shared_ptr<hypercube> slow_hyper, 
    std::shared_ptr<paramObj> par, 
    std::shared_ptr<RefSampler> ref = nullptr,
    std::shared_ptr<WavefieldPool> wfld_pool = nullptr,
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
    CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream),
    _param(par) {
  
      initialize(domain, slow_hyper, par, wfld_pool);
      // for now only support PSPI propagator
      prop = std::make_shared<PSPI>(domain, slow_hyper, par, ref, model_vec, data_vec, _grid_, _block_, _stream_);
    };

  void set_background_model(std::shared_ptr<complex4DReg> slow) {
    _slow_ = slow;
  }

  virtual ~OneWay() {};

  void one_step_fwd(int iz, complex_vector* __restrict__ wfld);
  void one_step_adj(int iz, complex_vector* __restrict__ wfld);

  void save_slice(int iz, complex_vector* wfld) {
    _wfld_pool->save_slice(iz, wfld, _stream_);
  };

  void load_slice(int iz, complex_vector* wfld) {
    _wfld_pool->load_slice(iz, wfld, _stream_);
  };

  float get_compression_ratio() {
    size_t comp_size = _wfld_pool->get_compressed_size();
    size_t orig_size = getDomainSizeInBytes() * m_ax[3].n;
    return static_cast<float>(orig_size) / static_cast<float>(comp_size);
  }

  std::shared_ptr<OneStep> getPropagator() {
    if (!prop) 
      throw std::runtime_error("Propagator is not initialized. Please check the constructor.");
    
    return prop;
  }

  std::shared_ptr<complex4DReg> getSlow() {
  if (!_slow_) 
      throw std::runtime_error("Slowness model is not initialized. Please check the constructor.");
    return _slow_;
  }

  std::shared_ptr<paramObj> getPar() {
    if (!_param) 
      throw std::runtime_error("Parameter object is not initialized. Please check the constructor.");
    
    return _param;
  }


protected:
  std::vector<axis> m_ax;
  // need slowness for split step propagator
  std::shared_ptr<complex4DReg> _slow_;
  std::shared_ptr<paramObj> _param;

  std::shared_ptr<WavefieldPool> _wfld_pool;

  std::shared_ptr<OneStep> prop;
  std::string _tag; // Tag for the wavefield pool


private:
  void initialize(std::shared_ptr<hypercube> domain, std::shared_ptr<hypercube> slow_hyper, std::shared_ptr<paramObj> par, std::shared_ptr<WavefieldPool> wfld_pool) {
    auto ax = domain->getAxes();
    m_ax = slow_hyper->getAxes();

    if (!wfld_pool) {
      std::string id = "wfld";
      _wfld_pool = std::make_shared<WavefieldPool>(domain, par, id, m_ax[3].n);
    } else {
      _wfld_pool = wfld_pool;
    }

    // _compressed_wflds.resize(m_ax[3].n); // Resize to number of slices in z-direction
  }

};

class Downward : public OneWay {
public:
  Downward (const std::shared_ptr<hypercube>& domain, std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par,
  std::shared_ptr<WavefieldPool> wfld_pool = nullptr,
  std::string tag = "down",
  complex_vector* model = nullptr, complex_vector* data = nullptr, 
  dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  OneWay(domain, slow, par, wfld_pool, model, data, grid, block, stream) {
    _tag = tag;
  };

  Downward (const std::shared_ptr<hypercube>& domain, std::shared_ptr<hypercube> slow_hyper, 
    std::shared_ptr<paramObj> par,
    std::shared_ptr<RefSampler> ref = nullptr,
    std::shared_ptr<WavefieldPool> wfld_pool = nullptr,
    std::string tag = "down",
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  OneWay(domain, slow_hyper, par, ref, wfld_pool, model, data, grid, block, stream) {
    _tag = tag;
  };

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  
};

class Upward : public OneWay {
public:
  Upward (const std::shared_ptr<hypercube>& domain,
  std::shared_ptr<complex4DReg> slow, std::shared_ptr<paramObj> par,
  std::shared_ptr<WavefieldPool> wfld_pool = nullptr,
  std::string tag = "up",
  complex_vector* model = nullptr, complex_vector* data = nullptr, 
  dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  OneWay(domain, slow, par, wfld_pool, model, data, grid, block, stream) {
    _tag = tag;
  };

  Upward (const std::shared_ptr<hypercube>& domain,
    std::shared_ptr<hypercube> slow_hyper, 
    std::shared_ptr<paramObj> par,
    std::shared_ptr<RefSampler> ref = nullptr,
    std::shared_ptr<WavefieldPool> wfld_pool = nullptr,
    std::string tag = "up",
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid = 1, dim3 block = 1, cudaStream_t stream = 0) :
  OneWay(domain, slow_hyper, par, ref, wfld_pool, model, data, grid, block, stream) {
    _tag = tag;
  };

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);

};