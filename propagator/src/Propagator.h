#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <complex2DReg.h>
#include <OneWay.h>
#include <RefSampler.h>
#include <Injection.h>
#include <Reflect.h>
#include <future>
#include <unordered_set>

class Propagator : public CudaOperator<complex2DReg, complex2DReg> {
public:
    // Constructor for wavelet -> data propagation
    Propagator(const std::shared_ptr<hypercube>& domain,
               const std::shared_ptr<hypercube>& range,
               std::shared_ptr<hypercube> slow_hyper,
               std::shared_ptr<complex2DReg> wavelet,
               const std::vector<float>& sx,
               const std::vector<float>& sy,
               const std::vector<float>& sz,
               const std::vector<int>& s_ids,
               const std::vector<float>& rx,
               const std::vector<float>& ry,
               const std::vector<float>& rz,
               const std::vector<int>& r_ids,
               std::shared_ptr<paramObj> par,
               complex_vector* model = nullptr,
               complex_vector* data = nullptr, 
               dim3 grid = dim3(1,1,1), dim3 block = dim3(1,1,1), cudaStream_t stream = 0);

    // Virtual destructor
    virtual ~Propagator() {
      wfld_slice_gpu->~complex_vector();
      CHECK_CUDA_ERROR(cudaFree(wfld_slice_gpu));
    };

    void set_background_model(std::vector<std::shared_ptr<complex4DReg>> model);

    void forward(bool add, std::vector<std::shared_ptr<complex4DReg>> model, std::shared_ptr<complex2DReg> data);

    void cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data){
      throw std::runtime_error("Not implemented");
    };
	  void cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data){
      throw std::runtime_error("Not implemented");
    };

    std::pair<double, double> get_compression_ratio() {
      std::pair<double, double> ratio;
      ratio.first = down->get_compression_ratio();
      ratio.second = up->get_compression_ratio();

      return ratio;
    }

    std::shared_ptr<Downward> getDown() {return down;}
    std::shared_ptr<Upward> getUp() {return up;}
    std::shared_ptr<Reflect> getReflect() {return reflect;}
    std::shared_ptr<Injection> getInjSrc() {return inj_src;}
    std::shared_ptr<Injection> getInjRec() {return inj_rec;}
    std::shared_ptr<hypercube> getWfldSliceHyper() {return wfld_hyper;}
    std::shared_ptr<RefSampler> getRefSampler() {return ref;}
    std::shared_ptr<WavefieldPool> getWfldPool() {return wfld_pool;}
    std::string getRunId() { return _run_id; }

    complex_vector* get_wfld_slice_buffer() {return wfld_slice_gpu;}

	

protected:
    // Wavefield pool
    std::shared_ptr<WavefieldPool> wfld_pool;

    // One-way propagators
    std::shared_ptr<Downward> down;
    std::shared_ptr<Upward> up;
    std::shared_ptr<Reflect> reflect;

    // Injection operators
    std::shared_ptr<Injection> inj_src;
    std::shared_ptr<Injection> inj_rec;
    
    // Reference velocity sampler
    std::shared_ptr<RefSampler> ref;
    int look_ahead;
    int decomp_look_ahead;

    std::string _run_id;
    
private:
    std::vector<axis> ax;
    std::shared_ptr<hypercube> _slow_hyper;
    std::shared_ptr<hypercube> wfld_hyper;
    complex_vector* wfld_slice_gpu;
};