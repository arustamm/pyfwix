#pragma once

#include <CudaOperator.h>
#include <complex4DReg.h>
#include <OneWay.h>
#include <prop_kernels.cuh>
#include <memory>

namespace SEP {

class ImagingCondition : public CudaOperator<complex3DReg, complex4DReg> {
public:
    ImagingCondition(
        const std::shared_ptr<hypercube>& domain,
				const std::shared_ptr<hypercube>& range,
        std::shared_ptr<OneWay> oneway,
        complex_vector* model = nullptr, 
        complex_vector* data = nullptr, 
        dim3 grid = 1, 
        dim3 block = 1, 
        cudaStream_t stream = 0
    );

    virtual ~ImagingCondition() {
        _bg_wfld_slice->~complex_vector();
        CHECK_CUDA_ERROR(cudaFree(_bg_wfld_slice));
    };

    // Set the depth for imaging condition
    void set_depth(int iz);

    // Forward imaging condition: image += conj(source_wfld) * receiver_wfld
    void cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) override;

    // Adjoint imaging condition: receiver_wfld += source_wfld * image
    void cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) override;

private:
    std::shared_ptr<OneWay> _oneway;  // OneWay object
    IC_launcher launchIC;            // Kernel launcher for imaging condition
    Pad_launcher launch_pad;            // Kernel launcher for padding
    complex_vector* _bg_wfld_slice; // Background wavefield slice
};

} // namespace SEP