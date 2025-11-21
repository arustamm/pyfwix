#pragma once
#include <vector>
#include <floatHyper.h>
#include <complexHyper.h>
#include <complex_vector.h>

using namespace SEP;

template <class M, class D>
class CudaOperator
{
public:

using DomainType = M; 
using RangeType = D;

CudaOperator(const std::shared_ptr<hypercube>& domain, const std::shared_ptr<hypercube>& range, 
							dim3 grid=1, dim3 block=1, cudaStream_t stream = 0) 
: _grid_(grid), _block_(block), _stream_(stream) {
	setDomainRange(domain, range);
};

CudaOperator(const std::shared_ptr<hypercube>& domain, const std::shared_ptr<hypercube>& range, 
							complex_vector* model = nullptr, complex_vector* data = nullptr, 
							// std::shared_ptr<paramObj> launch_param
							dim3 grid=1, dim3 block=1, cudaStream_t stream = 0) 
: CudaOperator(domain, range, grid, block, stream) {
	
	if (model == nullptr) {
		model_alloc = true;
		model_vec = make_complex_vector(domain, _grid_, _block_, _stream_);
	}
	else model_vec = model;

	if (data == nullptr) {
		data_alloc = true;
		data_vec = make_complex_vector(range, _grid_, _block_, _stream_);
	}
	else data_vec = data;
	
	model_vec->set_stream(stream);
	data_vec->set_stream(stream);
};

virtual ~CudaOperator() {
	if (model_alloc) {
		model_vec->~complex_vector();
		CHECK_CUDA_ERROR(cudaFree(model_vec));
	}
	if (data_alloc) {
		data_vec->~complex_vector();
		CHECK_CUDA_ERROR(cudaFree(data_vec));
	}
};

cudaStream_t get_stream() const {return _stream_;};

void set_grid(dim3 grid) {_grid_ = grid;};
void set_block(dim3 block) {_block_ = block;};

virtual void cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) = 0;
virtual void cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) = 0;
virtual void cu_inverse(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
	throw std::runtime_error("cu_inverse not implemented in the derived class."); 
};

virtual void cu_forward(complex_vector* __restrict__ model) {
	throw std::runtime_error("cu_forward not implemented in the derived class."); 
};
virtual void cu_adjoint(complex_vector* __restrict__ data) {
	throw std::runtime_error("cu_adjoint not implemented in the derived class."); 
};

virtual void forward(bool add, std::shared_ptr<M>& model, std::shared_ptr<D>& data) {
	// pin the host memory
	CHECK_CUDA_ERROR(cudaHostRegister(model->getVals(), getDomainSizeInBytes(), cudaHostRegisterDefault));
	CHECK_CUDA_ERROR(cudaHostRegister(data->getVals(), getRangeSizeInBytes(), cudaHostRegisterDefault));

	if (add) {
		CHECK_CUDA_ERROR(cudaMemcpyAsync(data_vec->mat, data->getVals(), getRangeSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
	}
	else {
		data->zero();
	}
	
	CHECK_CUDA_ERROR(cudaMemcpyAsync(model_vec->mat, model->getVals(), getDomainSizeInBytes(), cudaMemcpyHostToDevice, _stream_));

	cu_forward(add, model_vec, data_vec);

	CHECK_CUDA_ERROR(cudaMemcpyAsync(data->getVals(), data_vec->mat, getRangeSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));

	// unpin the memory
	CHECK_CUDA_ERROR(cudaHostUnregister(model->getVals()));
	CHECK_CUDA_ERROR(cudaHostUnregister(data->getVals()));
};

// this is host-to-host function
virtual void adjoint(bool add, std::shared_ptr<M>& model, std::shared_ptr<D>& data) {
	// pin the host memory
	CHECK_CUDA_ERROR(cudaHostRegister(model->getVals(), getDomainSizeInBytes(), cudaHostRegisterDefault));
	CHECK_CUDA_ERROR(cudaHostRegister(data->getVals(), getRangeSizeInBytes(), cudaHostRegisterDefault));

	if (add) {
		CHECK_CUDA_ERROR(cudaMemcpyAsync(model_vec->mat, model->getVals(), getDomainSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
	}
	else {
		model->zero();
	}

	CHECK_CUDA_ERROR(cudaMemcpyAsync(data_vec->mat,data->getVals(), getRangeSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
	cu_adjoint(add, model_vec, data_vec);
	CHECK_CUDA_ERROR(cudaMemcpyAsync(model->getVals(),model_vec->mat, getDomainSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));

	// unpin the memory
	CHECK_CUDA_ERROR(cudaHostUnregister(model->getVals()));
	CHECK_CUDA_ERROR(cudaHostUnregister(data->getVals()));
};

// this is host-to-host function
void inverse(bool add, std::shared_ptr<M>& model, std::shared_ptr<D>& data) {
	// pin the host memory
	CHECK_CUDA_ERROR(cudaHostRegister(model->getVals(), getDomainSizeInBytes(), cudaHostRegisterDefault));
	CHECK_CUDA_ERROR(cudaHostRegister(data->getVals(), getRangeSizeInBytes(), cudaHostRegisterDefault));

	if (add) {
		CHECK_CUDA_ERROR(cudaMemcpyAsync(model_vec->mat, model->getVals(), getDomainSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
	}
	else {
		model->zero();
	}

	CHECK_CUDA_ERROR(cudaMemcpyAsync(data_vec->mat,data->getVals(), getRangeSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
	cu_inverse(add, model_vec, data_vec);
	CHECK_CUDA_ERROR(cudaMemcpyAsync(model->getVals(),model_vec->mat, getDomainSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));

	// unpin the memory
	CHECK_CUDA_ERROR(cudaHostUnregister(model->getVals()));
	CHECK_CUDA_ERROR(cudaHostUnregister(data->getVals()));
};

virtual void forward(std::shared_ptr<D>& data) {
	// pin the host memory
	CHECK_CUDA_ERROR(cudaHostRegister(data->getVals(), getRangeSizeInBytes(), cudaHostRegisterDefault));
	
	CHECK_CUDA_ERROR(cudaMemcpyAsync(data_vec->mat, data->getVals(), getRangeSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
	cu_forward(true, data_vec, data_vec);
	CHECK_CUDA_ERROR(cudaMemcpyAsync(data->getVals(), data_vec->mat, getRangeSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));

	// unpin the memory
	CHECK_CUDA_ERROR(cudaHostUnregister(data->getVals()));
};

// this is host-to-host function
virtual void adjoint(std::shared_ptr<M>& model) {
	// pin the host memory
	CHECK_CUDA_ERROR(cudaHostRegister(model->getVals(), getDomainSizeInBytes(), cudaHostRegisterDefault));
	
	CHECK_CUDA_ERROR(cudaMemcpyAsync(model_vec->mat, model->getVals(), getDomainSizeInBytes(), cudaMemcpyHostToDevice, _stream_));
	cu_adjoint(true, model_vec, model_vec);
	CHECK_CUDA_ERROR(cudaMemcpyAsync(model->getVals(), model_vec->mat, getDomainSizeInBytes(), cudaMemcpyDeviceToHost, _stream_));

	// unpin the memory
	CHECK_CUDA_ERROR(cudaHostUnregister(model->getVals()));
};

const std::shared_ptr<hypercube>& getDomain() const{
	return _domain;
}
const std::shared_ptr<hypercube>& getRange() const{
	return _range;
}
const int getDomainSize() const{
	return _domain->getN123();
}
const int getRangeSize() const{
	return _range->getN123();
}
const size_t getDomainSizeInBytes() const{
	return sizeof(cuFloatComplex)*_domain->getN123();
}
const size_t getRangeSizeInBytes() const{
	return sizeof(cuFloatComplex)*_range->getN123();
}

void setDomainRange(const std::shared_ptr<hypercube>& domain, const std::shared_ptr<hypercube>& range) {
	_domain = domain->clone();
	_range = range->clone();
}

std::pair<double, double> dotTest(bool verbose = false) {
	std::shared_ptr<M> m1 = std::make_shared<M>(getDomain());
	std::shared_ptr<D> d1 = std::make_shared<D>(getRange());
	auto _model = m1->clone();
	auto _data = d1->clone();
	_model->random();
	_data->random();

	this->forward(false, _model,d1);
	this->adjoint(false, m1,_data);

	// std::cerr << typeid(*this).name() << '\n';
	std::pair<double, double> err;
	err.first = std::abs(std::real(std::conj(_data->dot(d1))/_model->dot(m1)) -1.);
	if (verbose) {
		std::cout << "********** ADD = FALSE **********" << '\n';
		std::cout << "<m,A'd>: " << _model->dot(m1) << std::endl;
		std::cout << "<Am,d>: " << std::conj(_data->dot(d1)) << std::endl;
		std::cout << "Error: " << err.first << "\n";
		std::cout << "*********************************" << '\n';
	}
	
	this->forward(true, _model,d1);
	this->adjoint(true, m1,_data);
	err.second = std::abs(std::real(std::conj(_data->dot(d1))/_model->dot(m1)) -1.);

	if (verbose) {
		std::cout << "********** ADD = TRUE **********" << '\n';
		std::cout << "<m,A'd>: " << _model->dot(m1) << std::endl;
		std::cout << "<Am,d>: " << std::conj(_data->dot(d1)) << std::endl;
		std::cout << "Error: " << err.second << "\n";
		std::cout << "*********************************" << '\n';
	}
	
	return err;
};

complex_vector *model_vec, *data_vec; 

protected:
std::shared_ptr<hypercube> _domain;
std::shared_ptr<hypercube> _range;
dim3 _grid_, _block_;
bool model_alloc = false, data_alloc = false;
cudaStream_t _stream_;

};

