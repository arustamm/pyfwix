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

     _block_ = 256;
    _grid_ = (this->getDomainSize() + _block_.x - 1) / _block_.x;

    // Calculate size of one depth slice (nx * ny * nw)
    _size_ = domain->getAxis(1).n * domain->getAxis(2).n * domain->getAxis(3).n;

    // 1. Allocate GPU memory for Interpolation Maps
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_labels_low,  sizeof(int) * _size_));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_labels_high, sizeof(int) * _size_));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_weights,     sizeof(float) * _size_));
    
    // Note: You must update Selector_launcher constructor if it needs specific args
    // Assuming standard instantiation is fine here.
    launcher = Selector_launcher(&select_forward, _grid_, _block_, _stream_);
    };
    
    ~Selector() {
        // 2. Clean up GPU memory
        CHECK_CUDA_ERROR(cudaFree(d_labels_low));
        CHECK_CUDA_ERROR(cudaFree(d_labels_high));
        CHECK_CUDA_ERROR(cudaFree(d_weights));
    };

    void set_block(dim3 block) {
        _grid_ = (this->getDomainSize() + block.x - 1) / block.x;
        launcher.set_grid_block(_grid_, block);
    }

    // 3. New Setter: Copies all 3 maps from Host (RefSampler) to Device
    inline void set_ref_maps(int* __restrict__ h_low, int* __restrict__ h_high, float* __restrict__ h_weights) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_labels_low,  h_low,     sizeof(int) * _size_,   cudaMemcpyHostToDevice, _stream_));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_labels_high, h_high,    sizeof(int) * _size_,   cudaMemcpyHostToDevice, _stream_));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_weights,     h_weights, sizeof(float) * _size_, cudaMemcpyHostToDevice, _stream_));
    };

    inline void set_value(int value) {_value_ = value;}

    void cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
        if (!add) data->zero();
        // Pass the 3 device pointers to the launcher
        launcher.run_fwd(model, data, _value_, d_labels_low, d_labels_high, d_weights);
    };

    void cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
        if (!add) model->zero();
        // Adjoint uses same logic (interpolation weights don't change direction)
        launcher.run_fwd(data, model, _value_, d_labels_low, d_labels_high, d_weights);
    };

private:
    int _value_;
    int _size_;
    
    // GPU Pointers
    int *d_labels_low;
    int *d_labels_high;
    float *d_weights;
    
    Selector_launcher launcher;

};