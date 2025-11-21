#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <prop_kernels.cuh>

using namespace SEP;

class Selector : public CudaOperator<complex4DReg,complex4DReg>
{

public:

	Selector(const std::shared_ptr<hypercube>& domain, 
	complex_vector* model = nullptr, complex_vector* data = nullptr, 
	dim3 grid=1, dim3 block=1, cudaStream_t stream = 0) 
	: CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream) {
	// 	_grid_ = {32, 4, 4};
  // _block_ = {16, 16, 4};

	 _block_ = 128;
  _grid_ = (this->getDomainSize() + _block_.x - 1) / _block_.x;

		_size_ = domain->getAxis(1).n * domain->getAxis(2).n * domain->getAxis(3).n;
		CHECK_CUDA_ERROR(cudaMalloc((void **)&d_labels, sizeof(int)*_size_));
		launcher = Selector_launcher(&select_forward, _grid_, _block_, _stream_);
	};
	
	~Selector() {
		CHECK_CUDA_ERROR(cudaFree(d_labels));
	};

	void set_block(dim3 block) {
		_grid_ = (this->getDomainSize() + block.x - 1) / block.x;
		launcher.set_grid_block(_grid_, block);
	}

	inline void set_labels(int* __restrict__ labels) {
		// labels are 3D -- (x,y,w)
		CHECK_CUDA_ERROR(cudaMemcpyAsync(d_labels, labels, sizeof(int)*_size_, cudaMemcpyHostToDevice, _stream_));
	};
	inline void set_value(int value) {_value_ = value;}

	void cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
		if (!add) data->zero();
		launcher.run_fwd(model, data, _value_, d_labels);
	};
	void cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
		if (!add) model->zero();
		launcher.run_fwd(data, model, _value_, d_labels);
	};

private:
	int _value_;
	int _size_;
	int *d_labels;
	Selector_launcher launcher;

};

