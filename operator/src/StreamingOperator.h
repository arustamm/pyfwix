#pragma once
#include <vector>
#include <floatHyper.h>
#include <complexHyper.h>
#include <complex_vector.h>
#include "CudaOperator.h"
#include <cuda_runtime.h>

using namespace SEP;

template <class Operator, typename... Args>
class StreamingOperator : public CudaOperator<typename Operator::DomainType, typename Operator::RangeType>
{
public:

	StreamingOperator(const std::shared_ptr<hypercube>& domain, const std::shared_ptr<hypercube>& range, 
								Args&&... args,
								dim3 grid=1, dim3 block=1, int nstreams = 1) 
	: CudaOperator<typename Operator::DomainType, typename Operator::RangeType>(domain, range) {

		auto ax = domain->getAxes();
		// Calculate the base number of items per batch
		int base = ax.back().n / nstreams;
		// Calculate the remainder
		int remainder = ax.back().n % nstreams;
		// Create a vector to store the batches
		std::vector<axis> last_axes;
		for (int i=0; i < nstreams; ++i) {
			float o = ax.back().o + i*base*ax.back().d;
			last_axes.push_back(axis(base, o, ax.back().d));
		}
		// Distribute the remainder
		for (int i = 0; i < remainder; ++i) {
			last_axes[i].n++;
			last_axes[i+1].o += last_axes[i].d;
		}
				
		for (int i=0; i < nstreams; ++i) {
			// create subrange hypercube 
			ax.back() = last_axes[i];
			subrange.push_back(std::make_shared<hypercube>(ax));
			// and subdata
			data_view.push_back(this->data_vec->make_view());
			int offset = i * last_axes[i].n;
			this->data_vec->view_at(data_view[i], offset);
			// create a new stream
			cudaStream_t s;
			CHECK_CUDA_ERROR(cudaStreamCreate(&s));
			stream.push_back(s);
			// create operators
			ops.push_back(Operator(domain, subrange[i], std::forward<Args>(args)..., this->model_vec, data_view[i], grid, block, stream[i]));
		}
	}

	StreamingOperator(const std::shared_ptr<hypercube>& domain, 
								dim3 grid=1, dim3 block=1, int nstreams = 1) 
	: CudaOperator<typename Operator::DomainType, typename Operator::DomainType>(domain, domain, nullptr, nullptr, grid, block) {
				
		auto ax = domain->getAxes();
		// Calculate the base number of items per batch
		int base = ax.back().n / nstreams;
		// Calculate the remainder
		int remainder = ax.back().n % nstreams;
		// Create a vector to store the batches
		std::vector<axis> last_axes;
		for (int i=0; i < nstreams; ++i) {
			float o = ax.back().o + i*base*ax.back().d;
			last_axes.push_back(axis(base, o, ax.back().d));
		}
		// Distribute the remainder
		for (int i = 0; i < remainder; ++i) {
			last_axes[i].n++;
			last_axes[i+1].o += last_axes[i].d;
		}
		
		int start = 0;
		int end = 0;
		for (int i=0; i < nstreams; ++i) {
			// create subrange hypercube 
			ax.back() = last_axes[i];
			auto subrange = std::make_shared<hypercube>(ax);
			auto subdomain = std::make_shared<hypercube>(ax);
			// and subdata
			end += last_axes[i].n;
			data_view.push_back(this->data_vec->make_view(start, end));
			model_view.push_back(this->model_vec->make_view(start, end));
			start = end;
			// create a new stream
			cudaStream_t s;
			CHECK_CUDA_ERROR(cudaStreamCreate(&s));
			stream.push_back(s);
			// create operators
			ops.push_back(std::make_unique<Operator>(subdomain, model_view[i], data_view[i], grid, block, stream[i]));
		}
	}

	~StreamingOperator() {
		for (int i=0; i < stream.size(); ++i) {
			CHECK_CUDA_ERROR(cudaStreamDestroy(stream[i]));
			data_view[i]->~complex_vector();
    	CHECK_CUDA_ERROR(cudaFree(data_view[i]));
			model_view[i]->~complex_vector();
    	CHECK_CUDA_ERROR(cudaFree(model_view[i]));
		}
	};

	virtual void cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
		throw std::runtime_error("Should call the host forward function instead.");
	};
	virtual void cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
		throw std::runtime_error("Should call the host forward function instead.");
	};

	void forward(bool add, std::shared_ptr<typename Operator::DomainType>& model, 
						std::shared_ptr<typename Operator::RangeType>& data) {
		// pin the host memory
		CHECK_CUDA_ERROR(cudaHostRegister(model->getVals(), this->getDomainSizeInBytes(), cudaHostRegisterDefault));
		CHECK_CUDA_ERROR(cudaHostRegister(data->getVals(), this->getRangeSizeInBytes(), cudaHostRegisterDefault));
		if (!add) data->zero();
		int offset = 0;

		for (int i=0; i < stream.size(); ++i) {

			if (add) {
				CHECK_CUDA_ERROR(cudaMemcpyAsync(data_view[i]->mat, data->getVals()+offset, ops[i]->getRangeSizeInBytes(), cudaMemcpyHostToDevice, stream[i]));
			}

			CHECK_CUDA_ERROR(cudaMemcpyAsync(model_view[i]->mat, model->getVals()+offset, ops[i]->getDomainSizeInBytes(), cudaMemcpyHostToDevice, stream[i]));

			ops[i]->cu_forward(add, model_view[i], data_view[i]);

			CHECK_CUDA_ERROR(cudaMemcpyAsync(data->getVals()+offset, data_view[i]->mat, ops[i]->getRangeSizeInBytes(), cudaMemcpyDeviceToHost, stream[i]));
			
			offset += ops[i]->getRangeSize();
		}

		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		// unpin the memory
		CHECK_CUDA_ERROR(cudaHostUnregister(model->getVals()));
		CHECK_CUDA_ERROR(cudaHostUnregister(data->getVals()));
	};

	void adjoint(bool add, std::shared_ptr<typename Operator::DomainType>& model, 
						std::shared_ptr<typename Operator::RangeType>& data) {
		// pin the host memory
		CHECK_CUDA_ERROR(cudaHostRegister(model->getVals(), this->getDomainSizeInBytes(), cudaHostRegisterDefault));
		CHECK_CUDA_ERROR(cudaHostRegister(data->getVals(), this->getRangeSizeInBytes(), cudaHostRegisterDefault));
		if (!add) model->zero();
		int offset = 0;

		for (int i=0; i < stream.size(); ++i) {

			if (add) {
				CHECK_CUDA_ERROR(cudaMemcpyAsync(model_view[i]->mat, model->getVals()+offset, ops[i]->getDomainSizeInBytes(), cudaMemcpyHostToDevice, stream[i]));
			}

			CHECK_CUDA_ERROR(cudaMemcpyAsync(data_view[i]->mat, data->getVals()+offset, ops[i]->getRangeSizeInBytes(), cudaMemcpyHostToDevice, stream[i]));

			ops[i]->cu_adjoint(add, model_view[i], data_view[i]);

			CHECK_CUDA_ERROR(cudaMemcpyAsync(model->getVals()+offset, model_view[i]->mat, ops[i]->getDomainSizeInBytes(), cudaMemcpyDeviceToHost, stream[i]));
			
			offset += ops[i]->getDomainSize();
		}

		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		// unpin the memory
		CHECK_CUDA_ERROR(cudaHostUnregister(model->getVals()));
		CHECK_CUDA_ERROR(cudaHostUnregister(data->getVals()));
	};

private:
	std::vector<cudaStream_t> stream;
	std::vector<std::unique_ptr<Operator>> ops;
	std::vector<hypercube> subrange, subdomain;
	std::vector<complex_vector*> data_view, model_view;
};

