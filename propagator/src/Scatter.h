#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <prop_kernels.cuh>
#include <FFT.h>
#include <paramObj.h>

using namespace SEP;

class Scatter : public CudaOperator<complex4DReg, complex4DReg> {
public:

Scatter(
    const std::shared_ptr<hypercube>& domain, 
    std::shared_ptr<complex4DReg> slow,
    std::shared_ptr<paramObj> par,
    complex_vector* model = nullptr, complex_vector* data = nullptr, 
    dim3 grid=1, dim3 block=1, cudaStream_t stream = 0);

    void cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
    void cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
    void cu_forward(complex_vector* __restrict__ model);
    void cu_adjoint(complex_vector* __restrict__ data);

   	void set_depth(int iz);


    virtual void set_grid_block(dim3 grid, dim3 block);

    ~Scatter() {
        CHECK_CUDA_ERROR(cudaFree(d_w));
        CHECK_CUDA_ERROR(cudaFree(d_kx));
        CHECK_CUDA_ERROR(cudaFree(d_ky));
        _slow_slice->~complex_vector();
        CHECK_CUDA_ERROR(cudaFree(_slow_slice));
        _padded_slow_slice->~complex_vector();
        CHECK_CUDA_ERROR(cudaFree(_padded_slow_slice));
        _wfld_k->~complex_vector();
        CHECK_CUDA_ERROR(cudaFree(_wfld_k));
        _wfld_scaled->~complex_vector();
        CHECK_CUDA_ERROR(cudaFree(_wfld_scaled));
    }

protected:
    // FFT object for 2D spatial transforms
    std::unique_ptr<cuFFT2d> fft2d;
    
    // Kernel launchers
    Mult_kxky launch_mult_kxky;
    Slow_scale launch_scale_by_slow;
    Scale_by_iw launch_scale_by_iw;
		Pad_launcher launch_pad;

    // Device arrays
    complex_vector* _slow_slice;
		complex_vector* _padded_slow_slice;
    float *d_w, *d_kx, *d_ky;
    
    // Temporary wavefields
    complex_vector* _wfld_k;
    complex_vector* _wfld_scaled;
    
    // Parameters
    float _dz_;
    float _eps_;
    int _ntaylor;
    std::shared_ptr<complex4DReg> _slow_;
    size_t _slice_size;
    
    // Taylor series coefficients
	std::vector<float> coef = {1., 1./2. , 3./8. , 5./16. , 35./128.};

    float* fill_in_k(const axis& ax) {
        float *k;
        CHECK_CUDA_ERROR(cudaMalloc((void **)&k, sizeof(float)*ax.n));
        auto h_k = std::vector<float>(ax.n);
        int n_half = ax.n / 2;
        float dk = 2*M_PI/(ax.d*ax.n);
        for (int ik = 0; ik <= n_half; ik++) {
            h_k[ik] = ik * dk;
        }
        for (int ik = n_half + 1; ik < ax.n; ik++) {
            h_k[ik] = (ik - ax.n) * dk;
        }
        CHECK_CUDA_ERROR(cudaMemcpyAsync(k, h_k.data(), sizeof(float)*ax.n, cudaMemcpyHostToDevice, _stream_));
        return k;
    }

    float* fill_in_w(const axis& ax) {
        float *w;
        CHECK_CUDA_ERROR(cudaMalloc((void **)&w, sizeof(float)*ax.n));
        auto h_w = std::vector<float>(ax.n);
        for (int i=0; i < h_w.size(); ++i) {
            float f = ax.o + i*ax.d;
            if (f == 0.f)
                throw std::runtime_error("Frequency is zero in the scattering operator!");
            h_w[i] = 2*M_PI*f;
        }
        CHECK_CUDA_ERROR(cudaMemcpyAsync(w, h_w.data(), sizeof(float)*ax.n, cudaMemcpyHostToDevice, _stream_));
        return w;
    }
};
