#pragma once
#include <cufft.h>
#include <cufftXt.h>
#include "CudaOperator.h"
#include "fft_callback.cuh"
#include <complex4DReg.h>

using namespace SEP;

class cuFFT2d : public CudaOperator<complex4DReg, complex4DReg> {
	public:
		cuFFT2d(const std::shared_ptr<hypercube>& domain, complex_vector* model = nullptr, complex_vector* data = nullptr, 
		dim3 grid = 1, dim3 block = 1,
		cudaStream_t stream = 0);
		
		~cuFFT2d() {
			temp->~complex_vector();
			CHECK_CUDA_ERROR(cudaFree(temp));
			CHECK_CUDA_ERROR(cudaFree(d_SIZE));
			cufftDestroy(plan);
		};

		// this is on-device functions
		void cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
		void cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
		void cu_forward(complex_vector* data);
		void cu_adjoint(complex_vector* data);

	private:
		cufftHandle plan;
		int NX, NY, BATCH, SIZE;
		int* d_SIZE; 
		complex_vector* temp;
		

};
