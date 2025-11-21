
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
	
	_grid_ = {32, 4, 4};
  	_block_ = {16, 16, 4};

	_bg_wfld_slice = data_vec->cloneSpace();
	launchIC = IC_launcher(&ic_fwd, &ic_adj, _grid_, _block_, _stream_);
}

void ImagingCondition::set_depth(int iz) {
	auto bg_wfld = _oneway->get_next_wfld_slice();
	CHECK_CUDA_ERROR(cudaMemcpyAsync(
		_bg_wfld_slice->mat,
		bg_wfld->getVals(), 
		getRangeSizeInBytes(), 
		cudaMemcpyHostToDevice, _stream_
	));
}


void ImagingCondition::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) data->zero();

	launchIC.run_fwd(model, data, _bg_wfld_slice);

}

void ImagingCondition::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {

	if(!add) model->zero();

  launchIC.run_adj(model, data, _bg_wfld_slice);

}

