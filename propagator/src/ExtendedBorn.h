#pragma once

#include <CudaOperator.h>
#include <complex4DReg.h>
#include <complex3DReg.h>
#include <complex2DReg.h>
#include <Propagator.h>
#include <Scattering.h>  // Assuming this contains DownScattering and UpScattering
#include <memory>
#include <vector>

namespace SEP {

class ExtendedBorn : public CudaOperator<complex4DReg, complex2DReg> {
public:
    ExtendedBorn(
        const std::shared_ptr<hypercube>& domain,
        const std::shared_ptr<hypercube>& range,
        const std::vector<std::shared_ptr<complex4DReg>>& slow_den,
        std::shared_ptr<Propagator> propagator,
        dim3 grid = 1,
        dim3 block = 1,
        cudaStream_t stream = 0
    );

    virtual ~ExtendedBorn() {
      dmodel->~complex_vector();
      CHECK_CUDA_ERROR(cudaFree(dmodel));
      dslow->~complex_vector();
      CHECK_CUDA_ERROR(cudaFree(dslow));
      CHECK_CUDA_ERROR(cudaHostUnregister(_slow->getVals()));
      CHECK_CUDA_ERROR(cudaHostUnregister(_den->getVals()));
      CHECK_CUDA_ERROR(cudaHostUnregister(local_data->getVals()));
      CHECK_CUDA_ERROR(cudaHostUnregister(hslow->getVals()));
      CHECK_CUDA_ERROR(cudaHostUnregister(hmodel->getVals()));
    };

    // Forward operator: extended Born modeling
    void forward(
        bool add,
        std::vector<std::shared_ptr<complex4DReg>> model,
        std::shared_ptr<complex2DReg> data
    );

    // Adjoint operator: extended Born imaging
    void adjoint(
        bool add,
        std::vector<std::shared_ptr<complex4DReg>> model,
        std::shared_ptr<complex2DReg> data
    );

    void cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data){
      throw std::runtime_error("Not implemented");
    };
	  void cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data){
      throw std::runtime_error("Not implemented");
    };

    void set_background_model(std::vector<std::shared_ptr<complex4DReg>> model);

    std::pair<double, double> dotTest(bool verbose=false) {
      // This dot test is specific to the ExtendedBorn operator, as it handles
      // a vector of model parameters (slowness and density).

      // 1. Create random model and data vectors
      // Model vector [slowness, density]
      auto m_slowness = std::make_shared<complex4DReg>(getDomain());
      auto m_density = std::make_shared<complex4DReg>(getDomain());
      m_slowness->random();
      m_density->random();
      std::vector<std::shared_ptr<complex4DReg>> m1 = {m_slowness, m_density};

      // Data vector
      auto d1 = std::make_shared<complex2DReg>(getRange());
      d1->random();

      // 2. Allocate space for the results of the forward and adjoint operations
      std::vector<std::shared_ptr<complex4DReg>> m2 = {
          std::make_shared<complex4DReg>(getDomain()),
          std::make_shared<complex4DReg>(getDomain())
      };
      auto d2 = std::make_shared<complex2DReg>(getRange());

      // 3. Perform the dot product test for add=false
      // Apply forward operator: d2 = A * m1
      this->forward(false, m1, d2);
      
      // Apply adjoint operator: m2 = A' * d1
      this->adjoint(false, m2, d1);

      // 4. Compute the dot products
      // <A*m1, d1>
      std::complex<double> fwd_dot = d2->dot(d1);
      // <m1, A'*d1>
      std::complex<double> adj_dot = m1[0]->dot(m2[0]) + m1[1]->dot(m2[1]);
      // 5. Calculate and display the error
      std::pair<double, double> err;
      err.first = std::abs(std::real(fwd_dot / adj_dot) - 1.0);

      if (verbose) {
          std::cout << "\n********** DOT PRODUCT TEST (add=false) **********" << '\n';
          std::cout << "<Am, d> (Forward dot product) : " << fwd_dot << std::endl;
          std::cout << "<m, A'd> (Adjoint dot product) : " << adj_dot << std::endl;
          std::cout << "Relative Error                  : " << err.first << "\n";
          std::cout << "**************************************************" << '\n';
      }

      // The add=true test 
      this->forward(true, m1, d2);
      // Apply adjoint operator: m2 = A' * d1
      this->adjoint(true, m2, d1);

      // <A*m1, d1>
      fwd_dot = d2->dot(d1);
      // <m1, A'*d1>
      adj_dot = m1[0]->dot(m2[0]) + m1[1]->dot(m2[1]);
      err.second = std::abs(std::real(fwd_dot / adj_dot) - 1.0);

      if (verbose) {
          std::cout << "\n********** DOT PRODUCT TEST (add=true) **********" << '\n';
          std::cout << "<Am, d> (Forward dot product) : " << fwd_dot << std::endl;
          std::cout << "<m, A'd> (Adjoint dot product) : " << adj_dot << std::endl;
          std::cout << "Relative Error                  : " << err.second << "\n";
          std::cout << "**************************************************" << '\n';
      }

      return err;
  }

private:

    void downward_scattering_fwd(const std::vector<std::shared_ptr<complex4DReg>>& model);
    void downward_scattering_adj(std::vector<std::shared_ptr<complex4DReg>>& model);
    
    void downward_reflected_scattering_fwd(const std::vector<std::shared_ptr<complex4DReg>>& model);
    void downward_reflected_scattering_adj(std::vector<std::shared_ptr<complex4DReg>>& model);

    void upward_scattering_fwd(const std::vector<std::shared_ptr<complex4DReg>>& model);
    void upward_scattering_adj(std::vector<std::shared_ptr<complex4DReg>>& model);
    
    void backward_scattering_fwd(const std::vector<std::shared_ptr<complex4DReg>>& model);
    void backward_scattering_adj(std::vector<std::shared_ptr<complex4DReg>>& model);


    // Get the size of a single depth slice
    size_t getSliceSize() const;

    // Get the size of a single depth slice in bytes
    size_t getSliceSizeInBytes() const;

    // Propagation operators
    std::shared_ptr<Downward> down, bg_down;           // Downward propagator
    std::shared_ptr<Upward> up, bg_up;             // Upward propagator
    std::shared_ptr<Reflect> bg_reflect;        // Reflection operator
    std::shared_ptr<Injection> inj_rec;        // Injection/recording operator
    std::shared_ptr<Propagator> _propagator; // Propagator instance
    std::shared_ptr<complex4DReg> _slow;     // Slowness model
    std::shared_ptr<complex4DReg> _den;      // Density model

    // Scattering operators
    std::shared_ptr<DownScattering> down_scattering;    // Downward scattering
    std::shared_ptr<UpScattering> up_scattering;        // Upward scattering
    std::shared_ptr<BackScattering> back_scattering;        // Back scattering

    // Working arrays
    std::shared_ptr<complex3DReg> hslow;            // Current slowness slice on host
    std::shared_ptr<complex5DReg> hmodel;           // Current model slice on host
    std::shared_ptr<complex2DReg> local_data;       // Local data slice on host
    complex_vector* dmodel;            // Current slowness slice on device
    complex_vector* dslow;             // Current slowness slice on device
    complex_vector* dden;
    complex_vector* wfld_slice_gpu;
    std::vector<axis> ax;

    
};

} // namespace SEP