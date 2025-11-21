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
  
  void compress_slice(int iz, complex_vector* __restrict__ wfld) {
    
    _wfld_pool->check_ready();
    _wfld_pool->compress_slice_async(iz, wfld, _stream_, _tag);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(_stream_));
    
  }

  void check_ready() { _wfld_pool->check_ready(); }
  void wait_to_finish() { _wfld_pool->wait_to_finish(); }
  int get_decomp_queue_size() {return decomp_queue.size(); }

  double get_compression_ratio() const {
    double compressed_size = static_cast<double>(_wfld_pool->get_total_compressed_size());
    double original_total_size = static_cast<double>(getDomainSizeInBytes() * m_ax[3].n);
    
    if (compressed_size == 0) return 0.0;  // Avoid division by zero
    
    return original_total_size / compressed_size;
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

  std::shared_ptr<complex4DReg> get_next_wfld_slice() {
    // This function now gets the NEXT available wavefield from the pipeline.
    if (decomp_queue.empty()) {
      throw std::runtime_error("The decompression queue is empty!");
    }

    // 1. Get the oldest request (the buffer index and its future) from the queue.
    auto& [buffer_idx, future] = decomp_queue.front();

    // 2. Wait for the decompression to finish and get the result.
    std::shared_ptr<complex4DReg> wfld_slice = future.get();

    // 3. Now that we have the result, remove the item from the queue.
    decomp_queue.pop();

    // 4. CRITICAL: Release the buffer back to the pool so it can be reused.
    _wfld_pool->release_decomp_buffer(buffer_idx);

    // 5. Return the ready-to-use wavefield.
    return wfld_slice;
  }

  void start_decompress_from_top() {
    // Schedule decompression for the first few slices before the loop starts.
    // if (_wfld_pool->get_pool_size() > 0) 
    //   throw std::runtime_error("Wavefield pool has not been clreared before initializing!");
    
    for (int i = 0; i < _wfld_pool->get_pool_size(); ++i) {
      int iz = i;
      if (iz > m_ax[3].n - 1) break;
      decomp_queue.push(this->decompress_slice_async(iz));
    }
  }

  void start_decompress_from_bottom() {
    // Schedule decompression for the first few slices before the loop starts.
    // if (_wfld_pool->get_pool_size() > 0) 
    //   throw std::runtime_error("Wavefield pool has not been clreared before initializing!");

    for (int i = 0; i < _wfld_pool->get_pool_size(); ++i) {
      int iz = m_ax[3].n - 1 - i;
      if (iz < 0) break;
      decomp_queue.push(this->decompress_slice_async(iz));
    }
  }

  void add_decompresss_from_top(int iz) {
    // Schedule the next decompression task to maintain the look-ahead window.
    // +1 because the results are popped from the queue before scheduling the next one.
		int next = _wfld_pool->get_pool_size() + iz ; 
    if (next < m_ax[3].n) 
      decomp_queue.push(this->decompress_slice_async(next));    
  }

  void add_decompresss_from_bottom(int iz) {
    // Schedule the next decompression task to maintain the look-ahead window.
		int next = iz - _wfld_pool->get_pool_size(); 
    if (next >= 0) 
      decomp_queue.push(this->decompress_slice_async(next));    
  }


protected:
  std::vector<axis> m_ax;
  // need slowness for split step propagator
  std::shared_ptr<complex4DReg> _slow_;
  std::shared_ptr<paramObj> _param;

  std::shared_ptr<WavefieldPool> _wfld_pool;
  // std::vector<std::vector<char>> _compressed_wflds;

  std::shared_ptr<OneStep> prop;
  // Futures for decompression of wavefields
  std::queue<std::pair<int, std::future<std::shared_ptr<complex4DReg>>>> decomp_queue;
  std::string _tag; // Tag for the wavefield pool


private:
  void initialize(std::shared_ptr<hypercube> domain, std::shared_ptr<hypercube> slow_hyper, std::shared_ptr<paramObj> par, std::shared_ptr<WavefieldPool> wfld_pool) {
    auto ax = domain->getAxes();
    m_ax = slow_hyper->getAxes();

    if (!wfld_pool) {
      std::string id = "wfld";
      _wfld_pool = std::make_shared<WavefieldPool>(domain, par, id);
    } else {
      _wfld_pool = wfld_pool;
    }

    // _compressed_wflds.resize(m_ax[3].n); // Resize to number of slices in z-direction
  }

  std::pair<int, std::future<std::shared_ptr<complex4DReg>>> decompress_slice_async(int iz) {
    return _wfld_pool->decompress_slice_async(iz, _tag);
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