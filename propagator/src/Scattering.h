#pragma once

#include <CudaOperator.h>
#include <complex4DReg.h>
#include <OneWay.h>
#include <ImagingCondition.h>
#include <Scatter.h>
#include <LinReflect.h>
#include <memory>

namespace SEP {

class ForwardScattering : public CudaOperator<complex3DReg, complex4DReg> {
public:
    ForwardScattering(
        const std::shared_ptr<hypercube>& domain,
        const std::shared_ptr<hypercube>& range,
        const std::shared_ptr<complex4DReg>& slow,
        std::shared_ptr<OneWay> oneway,
        complex_vector* model = nullptr, 
        complex_vector* data = nullptr, 
        dim3 grid = 1, 
        dim3 block = 1, 
        cudaStream_t stream = 0
    );

    virtual ~ForwardScattering() {
        sc_wfld->~complex_vector();
        CHECK_CUDA_ERROR(cudaFree(sc_wfld));
    };

    // Set the depth for all operators (imaging condition, scatter, propagator)
    virtual void set_depth(int iz);

    // Forward scattering: model -> scattered wavefield -> propagated data
    void cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);

    // Adjoint scattering: data -> back-propagated -> adjoint scatter -> adjoint imaging -> model
    void cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);

protected:
    std::shared_ptr<Scatter> sc;                // Scattering operator
    std::shared_ptr<OneStep> prop;              // Propagator from OneWay
    std::shared_ptr<ImagingCondition> ic;       // Imaging condition operator
    complex_vector* sc_wfld;                // Temporary vector for intermediate results
};

class DownScattering : public ForwardScattering {
public:
    DownScattering(
        const std::shared_ptr<hypercube>& domain,
        const std::shared_ptr<hypercube>& range,
        const std::shared_ptr<complex4DReg>& slow,
        std::shared_ptr<Downward> oneway,
        complex_vector* model = nullptr, 
        complex_vector* data = nullptr, 
        dim3 grid = 1, 
        dim3 block = 1, 
        cudaStream_t stream = 0
    ) : ForwardScattering(domain, range, slow, oneway, model, data, grid, block, stream) {};
};

class UpScattering : public ForwardScattering {
public:
    UpScattering(
        const std::shared_ptr<hypercube>& domain,
        const std::shared_ptr<hypercube>& range,
        const std::shared_ptr<complex4DReg>& slow,
        std::shared_ptr<Upward> oneway,
        complex_vector* model = nullptr, 
        complex_vector* data = nullptr, 
        dim3 grid = 1, 
        dim3 block = 1, 
        cudaStream_t stream = 0
    ) : ForwardScattering(domain, range, slow, oneway, model, data, grid, block, stream) {};

    void set_depth(int iz) override;
};

class BackScattering : public CudaOperator<complex5DReg, complex4DReg> {
public:
    BackScattering(
        const std::shared_ptr<hypercube>& domain,
        const std::shared_ptr<hypercube>& range,
	    const std::vector<std::shared_ptr<complex4DReg>>& slow_impedance,
        std::shared_ptr<OneWay> oneway,
        complex_vector* model = nullptr, 
        complex_vector* data = nullptr, 
        dim3 grid = 1, 
        dim3 block = 1, 
        cudaStream_t stream = 0
    );

    ~BackScattering() {
        sc_wfld->~complex_vector();
        CHECK_CUDA_ERROR(cudaFree(sc_wfld));
    };

    void set_depth(int iz);
    // Forward scattering: model -> scattered wavefield -> propagated data
    void cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);
    // Adjoint scattering: data -> back-propagated -> adjoint scatter -> adjoint imaging -> model
    void cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data);

private:
    std::shared_ptr<OneStep> prop;              // Propagator from OneWay
    std::shared_ptr<ImagingCondition> ic;       // Imaging condition operator
    std::shared_ptr<LinReflect> lin_refl;       // Linearized reflection operator
    complex_vector* sc_wfld;                // Temporary vector for intermediate results
};

} // namespace SEP