#include "Reflect.h"

BaseReflect::BaseReflect (const std::shared_ptr<hypercube>& domain, 
  const std::vector<std::shared_ptr<complex4DReg>>& slow_impedance, 
  complex_vector* model, complex_vector* data, 
  dim3 grid, dim3 block, cudaStream_t stream) :
CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream) {
  initialize(slow_impedance[0]->getHyper());
  set_background_model(slow_impedance);
};

BaseReflect::BaseReflect (const std::shared_ptr<hypercube>& domain, std::shared_ptr<hypercube> slow_hyper, 
  complex_vector* model, complex_vector* data, 
  dim3 grid, dim3 block, cudaStream_t stream) :
CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream) {
  initialize(slow_hyper);
};

void BaseReflect::initialize(std::shared_ptr<hypercube> slow_hyper) {
  _grid_ = {32, 4, 4};
  _block_ = {16, 16, 4};

  nz = slow_hyper->getAxis(4).n;;
  nw = slow_hyper->getAxis(3).n;
  ny = slow_hyper->getAxis(2).n;
  nx = slow_hyper->getAxis(1).n;

  d_slow_slice = make_complex_vector(std::make_shared<hypercube>(nx, ny, nw, 2), _grid_, _block_, _stream_);
  d_den_slice = make_complex_vector(std::make_shared<hypercube>(nx, ny, nw, 2), _grid_, _block_, _stream_);
  
  slice_size = nx * ny * nw;
}



void Reflect::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) data->zero();
  launcher.run_fwd(model, data, d_slow_slice, d_den_slice);

}

void Reflect::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
	if(!add) model->zero();
  launcher.run_adj(model, data, d_slow_slice, d_den_slice);
}

void Reflect::cu_forward(complex_vector* __restrict__ model) {
  launcher_in_place.run_fwd(model, model, d_slow_slice, d_den_slice);
}

void Reflect::cu_adjoint(complex_vector* __restrict__ data) {
  launcher_in_place.run_adj(data, data, d_slow_slice, d_den_slice);
}