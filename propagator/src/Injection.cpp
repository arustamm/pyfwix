#include "Injection.h"
#include <prop_kernels.cuh>
#include <cuda.h>

Injection::Injection(const std::shared_ptr<hypercube>& domain,const std::shared_ptr<hypercube>& range, 
  float oz, float dz,
  complex_vector* model, complex_vector* data, dim3 grid, dim3 block, cudaStream_t stream) 
: CudaOperator<complex2DReg, complex4DReg>(domain, range, model, data, grid, block, stream) {

  this->oz = oz;
  this->dz = dz;
  
  _grid_ = {32, 16};
  _block_ = {32, 16};

  launcher = Injection_launcher(&inj_forward, &inj_adjoint, _grid_, _block_, _stream_);
  
  ntrace = domain->getAxis(2).n; // sources or receivers

  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_cx, ntrace * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_cy, ntrace * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_cz, ntrace * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_ids, ntrace * sizeof(int)));
};

Injection::Injection(const std::shared_ptr<hypercube>& domain,const std::shared_ptr<hypercube>& range, 
float oz, float dz,
const std::vector<float>& cx, const std::vector<float>& cy, const std::vector<float>& cz, const std::vector<int>& ids, 
complex_vector* model, complex_vector* data, dim3 grid, dim3 block, cudaStream_t stream) 
: Injection(domain, range, oz, dz, model, data, grid, block, stream) {
  
  set_coords(cx, cy, cz, ids);

};


  
void Injection::cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
  if (!add) data->zero();
  launcher.run_fwd(model, data, d_cx, d_cy, d_cz, d_ids, this->oz, this->dz, this->iz);

};
void Injection::cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
  if (!add) model->zero();
  launcher.run_adj(model, data, d_cx, d_cy, d_cz, d_ids, this->oz, this->dz, this->iz);
};
