#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <prop_kernels.cuh>

namespace SEP {

class Taper : public CudaOperator<complex4DReg,complex4DReg>
{
public:
	Taper(const std::shared_ptr<hypercube>& domain, std::shared_ptr<paramObj> par,
	complex_vector* model_vec=nullptr, complex_vector* data_vec=nullptr,
	dim3 grid=1, dim3 block=1, cudaStream_t stream=0
): CudaOperator<complex4DReg,complex4DReg>(domain,domain,model_vec,data_vec,grid,block,stream) {

		tapx = par->getInt("taperx", 0);
		tapy = par->getInt("tapery", 0);

		_grid_ = {32, 4, 4};
  	_block_ = {16, 16, 4};

		launcher = Taper_launcher(&taper_forward, _grid_, _block_, _stream_);

	};

	void cu_forward (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
		if (!add) data->zero();
		launcher.run_fwd(model, data, tapx, tapy);
	};
	void cu_adjoint (bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
		if (!add) model->zero();
		launcher.run_fwd(data, model, tapx, tapy);
	};

private:
	int tapx, tapy;
	Taper_launcher launcher;
};

}
