#pragma once

#include <CudaOperator.h>
#include <complex4DReg.h>
#include <complex3DReg.h>
#include <complex2DReg.h>
#include <Propagator.h>
#include <ImagingCondition.h> 
#include <memory>
#include <vector>

namespace SEP {

class ExtendedMigration : public CudaOperator<complex4DReg, complex2DReg> {
public:
    ExtendedMigration(
        const std::shared_ptr<hypercube>& domain,
        const std::shared_ptr<hypercube>& range,
        const std::vector<std::shared_ptr<complex4DReg>>& slow_den,
        std::shared_ptr<Propagator> propagator,
        dim3 grid = 1,
        dim3 block = 1,
        cudaStream_t stream = 0
    );

    virtual ~ExtendedMigration() {
      dimage->~complex_vector();
      CHECK_CUDA_ERROR(cudaFree(dimage));
    //   CHECK_CUDA_ERROR(cudaHostUnregister(himage->getVals()));
    };

    // Just migrate
    void migrate(
        bool add,
        std::shared_ptr<complex4DReg> image,
        std::shared_ptr<complex2DReg> data
    );

    void cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data){
      throw std::runtime_error("Not implemented");
    };
	  void cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data){
      throw std::runtime_error("Not implemented");
    };

private:

    // Get the size of a single depth slice
    size_t getSliceSize() const;

    // Get the size of a single depth slice in bytes
    size_t getSliceSizeInBytes() const;

    // Propagation operators
    std::shared_ptr<Downward> bg_down;           // Downward propagator
    std::shared_ptr<Upward> up;             // Upward propagator
    std::shared_ptr<Injection> inj_rec;        // Injection/recording operator
    std::shared_ptr<Propagator> _propagator; // Propagator instance
    std::shared_ptr<complex4DReg> _slow;     // Slowness model
    std::shared_ptr<complex4DReg> _den;      // Density model

    std::shared_ptr<ImagingCondition> ic; // Imaging condition

    // Working arrays
    std::shared_ptr<complex3DReg> himage;            // Current slowness slice on host
    complex_vector* dimage;            // Current slowness slice on device
    std::vector<axis> ax;

    
};

} // namespace SEP