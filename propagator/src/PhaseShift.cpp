#include "PhaseShift.h"
#include <prop_kernels.cuh>
#include <cuda.h>


PhaseShift::PhaseShift(const std::shared_ptr<hypercube>& domain, float dz, float eps, 
complex_vector* model, complex_vector* data, dim3 grid, dim3 block, cudaStream_t stream) 
: CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream), _dz_(dz), _eps_(eps) {

  _grid_ = {32, 4, 4};
  _block_ = {16, 16, 4};

  // _block_ = 256;
  // _grid_ = (this->getDomainSize() + _block_.x - 1) / _block_.x;

  launcher = PS_launcher(&ps_forward, &ps_adjoint, _grid_, _block_, _stream_);
  launcher_inv = PS_launcher(&ps_forward, &ps_inverse, _grid_, _block_, _stream_); 

  d_w2 = fill_in_w(domain->getAxis(3));
  d_ky = fill_in_k(domain->getAxis(2));
  d_kx = fill_in_k(domain->getAxis(1));

  _nw_ = domain->getAxis(3).n;
  CHECK_CUDA_ERROR(cudaMalloc((void**)&_sref_, _nw_ * sizeof(std::complex<float>)));
};

void PhaseShift::set_grid_block(dim3 grid, dim3 block) {
  launcher.set_grid_block(grid, block);
  launcher_inv.set_grid_block(grid, block);
}

void PhaseShift::cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
  if (!add) data->zero();
  launcher.run_fwd(model, data, d_w2, d_kx, d_ky, _sref_, _dz_, _eps_);
};


void PhaseShift::cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
  if (!add) model->zero();
  launcher.run_adj(model, data, d_w2, d_kx, d_ky, _sref_, _dz_, _eps_);
}

void PhaseShift::cu_inverse (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
  if (!add) model->zero();
  launcher_inv.run_adj(model, data, d_w2, d_kx, d_ky, _sref_, _dz_, _eps_);
}
