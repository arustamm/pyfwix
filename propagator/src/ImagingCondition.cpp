
#include <ImagingCondition.h>

using namespace SEP;

ImagingCondition::ImagingCondition(
	const std::shared_ptr<hypercube>& domain,
	const std::shared_ptr<hypercube>& range,
	std::shared_ptr<OneWay> oneway,
	complex_vector* model, complex_vector* data, 
  dim3 grid, dim3 block, cudaStream_t stream
) : 
CudaOperator<complex3DReg, complex4DReg>(domain, range, model, data, grid, block, stream) ,
_oneway(oneway)	{
	
  	_block_ = {8, 8, 4};
	_grid_.x = (range->getAxis(1).n + _block_.x - 1) / _block_.x;
	_grid_.y = (range->getAxis(2).n + _block_.y - 1) / _block_.y;
	_grid_.z = (range->getAxis(3).n*range->getAxis(4).n + _block_.z - 1) / _block_.z;

	_bg_wfld_slice = data_vec->cloneSpace();
	launchIC = IC_launcher(&ic_fwd, &ic_adj, _grid_, _block_, _stream_);
}

void ImagingCondition::set_depth(int iz) {
	_oneway->load_slice(iz, _bg_wfld_slice);
}


void ImagingCondition::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) data->zero();

	launchIC.run_fwd(model, data, _bg_wfld_slice);

}

void ImagingCondition::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) model->zero();

  launchIC.run_adj(model, data, _bg_wfld_slice);

}

