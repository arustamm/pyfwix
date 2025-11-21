#include "LinReflect.h"

LinReflect::LinReflect (
  const std::shared_ptr<hypercube>& domain, 
  const std::shared_ptr<hypercube>& range, 
  const std::vector<std::shared_ptr<complex4DReg>>& slow_impedance, 
  complex_vector* model, complex_vector* data, 
  dim3 grid, dim3 block, cudaStream_t stream) :
CudaOperator<complex5DReg, complex3DReg>(domain, range, model, data, grid, block, stream) {

  set_background_model(slow_impedance);
  
  _grid_ = {32, 4, 4};
  _block_ = {16, 16, 4};
  
  auto slow_hyper = slow_impedance[0]->getHyper();

  nz = slow_hyper->getAxis(4).n;;
  nw = slow_hyper->getAxis(3).n;
  ny = slow_hyper->getAxis(2).n;
  nx = slow_hyper->getAxis(1).n;

  d_slow_slice = make_complex_vector(std::make_shared<hypercube>(nx, ny, nw, 2), _grid_, _block_, _stream_);
  d_den_slice = make_complex_vector(std::make_shared<hypercube>(nx, ny, nw, 2), _grid_, _block_, _stream_);
  
  slice_size = nx * ny * nw;

  launcher = Refl_launcher(&drefl_forward, &drefl_adjoint, _grid_, _block_, _stream_);
};

void LinReflect::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) data->zero();
  launcher.run_fwd(model, data, d_slow_slice, d_den_slice);

}

void LinReflect::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
	if(!add) model->zero();
  launcher.run_adj(model, data, d_slow_slice, d_den_slice);
}