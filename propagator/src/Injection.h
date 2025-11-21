#pragma once
#include "CudaOperator.h"
#include <complex4DReg.h>
#include <complex2DReg.h>
#include <prop_kernels.cuh>

using namespace SEP;

// operator injecting a wavelet or data into the set of wavefields: [Ns, Nw, Nx, Ny]
class Injection : public CudaOperator<complex2DReg, complex4DReg>  {
public:
  
  Injection(const std::shared_ptr<hypercube>& domain,const std::shared_ptr<hypercube>& range, 
    float oz, float dz,
    complex_vector* model = nullptr, complex_vector* data = nullptr, dim3 grid=1, dim3 block=1, cudaStream_t stream = 0);

  Injection(const std::shared_ptr<hypercube>& domain,const std::shared_ptr<hypercube>& range, 
    float oz, float dz,
    const std::vector<float>& cx, const std::vector<float>& cy, const std::vector<float>& cz, const std::vector<int>& ids, 
    complex_vector* model = nullptr, complex_vector* data = nullptr, dim3 grid=1, dim3 block=1, cudaStream_t stream = 0);
  
  ~Injection() {
    CHECK_CUDA_ERROR(cudaFree(d_cx));
    CHECK_CUDA_ERROR(cudaFree(d_cy));
    CHECK_CUDA_ERROR(cudaFree(d_cz));
    CHECK_CUDA_ERROR(cudaFree(d_ids));
  };

  void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
  void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);

  void set_depth(int iz) {
    this->iz = iz;
  };

  void set_coords(const std::vector<float>& cx, const std::vector<float>& cy, const std::vector<float>& cz, const std::vector<int>& ids) {
    set_coords(cx.data(), cy.data(), cz.data(), ids.data());
  };

   void set_coords(const float* cx, const float* cy, const float* cz, const int* ids) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_cx, cx, sizeof(float)*ntrace, cudaMemcpyHostToDevice, _stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_cy, cy, sizeof(float)*ntrace, cudaMemcpyHostToDevice, _stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_cz, cz, sizeof(float)*ntrace, cudaMemcpyHostToDevice, _stream_));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_ids, ids, sizeof(int)*ntrace, cudaMemcpyHostToDevice, _stream_));
  };

private:
  Injection_launcher launcher;
  float *d_cx, *d_cy, *d_cz;
  int *d_ids;
  const std::vector<float> _cx, _cy, _cz;
  const std::vector<int> _ids;
  int ntrace, iz;
  float oz, dz;
};
