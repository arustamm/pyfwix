#include <complex4DReg.h>
#include "fftw3.h"
#include <FFT.h>
#include <benchmark/benchmark.h>
#include <complex_vector.h>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <iomanip>
#include <StreamingOperator.h>

using namespace SEP;

class FFT2d {
public:
  FFT2d(const std::shared_ptr<hypercube>& domain, const std::shared_ptr<hypercube>& range) {
    NX = domain->getAxis(1).n;
    NY = domain->getAxis(2).n;
    BATCH = domain->getN123() / (NX*NY);
    SIZE = domain->getN123();  
    int dims[2] = {NY, NX};
		// similar to cumalloc in cufft
    fftwf_init_threads();
    int nthreads = std::thread::hardware_concurrency();
    fftwf_plan_with_nthreads(nthreads);
    _model = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*SIZE);
    _data = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*SIZE);

    _fwd_plan = fftwf_plan_many_dft(2, dims, BATCH,
	                             _model, NULL, 1, 0,
	                             _data, NULL, 1, 0,
	                             FFTW_FORWARD,FFTW_MEASURE );
    fftwf_free(_model);
    fftwf_free(_data);
  };

  ~FFT2d() {
		fftwf_destroy_plan(_fwd_plan);
		// fftwf_cleanup();
	};

  void forward (bool add, const std::shared_ptr<complex4DReg> model, std::shared_ptr<complex4DReg> data) {

	// if(!add) data->scale(0.);
  _model = (reinterpret_cast<fftwf_complex*>(model->getVals()));
	_data = (reinterpret_cast<fftwf_complex*>(data->getVals()));
	fftwf_execute_dft(_fwd_plan,_model,_data);

	data->scale(1./std::sqrt(NX*NY));

};

private:
  int NX, NY, BATCH, SIZE; 
  fftwf_plan _fwd_plan;
  fftwf_complex *_model, *_data;
};

// Benchmark function for CPU FFT
double benchmark_cpu_fft(int n1, int n2, int n3, int n4, int iterations) {
  auto hyper = std::make_shared<hypercube>(n1, n2, n3, n4);
  auto model = std::make_shared<complex4DReg>(hyper);
  auto data = std::make_shared<complex4DReg>(hyper);
  model->set(1.0f);
  
  auto fft = std::make_unique<FFT2d>(hyper, hyper);
  
  double total_time = 0.0;
  
  for (int i = 0; i < iterations; ++i) {
      auto start = std::chrono::high_resolution_clock::now();
      fft->forward(false, model, data);
      auto end = std::chrono::high_resolution_clock::now();
      
      std::chrono::duration<double> elapsed = end - start;
      total_time += elapsed.count();
  }
  
  return total_time / iterations;
}

double benchmark_gpu_fft_host(int n1, int n2, int n3, int n4, int iterations) {
  auto hyper = std::make_shared<hypercube>(n1, n2, n3, n4);
  auto model = std::make_shared<complex4DReg>(hyper);
  auto data = std::make_shared<complex4DReg>(hyper);
  model->set(1.0f);
  
  dim3 grid = {16, 16, 4};
    dim3 block = {16, 16, 4};
  auto fft = std::make_unique<cuFFT2d>(hyper, nullptr, nullptr, grid, block);
  
  double total_time = 0.0;
  
  for (int i = 0; i < iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    fft->forward(false, model, data);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    total_time += elapsed.count();
  }
  
  return total_time / iterations;
}

// Benchmark function for GPU FFT (device only)
double benchmark_gpu_fft_device(int n1, int n2, int n3, int n4, int iterations) {
    auto hyper = std::make_shared<hypercube>(n1, n2, n3, n4);
    auto model = std::make_shared<complex4DReg>(hyper);
    auto data = std::make_shared<complex4DReg>(hyper);
    model->set(1.0f);
    
    dim3 grid = {16, 16, 4};
    dim3 block = {16, 16, 4};
    auto fft = std::make_unique<cuFFT2d>(hyper, nullptr, nullptr, grid, block);
    
    double total_time = 0.0;
    
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        fft->cu_forward(false, fft->model_vec, fft->data_vec);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        total_time += elapsed.count();
    }
    
    return total_time / iterations;
}

int main() {
  // Test configurations
  std::vector<std::vector<int>> configs = {
      {1, 1, 1000, 1000},
      {1, 10, 1000, 1000},
      {1, 100, 1000, 1000},
      {1, 1000, 1000, 1000},
      {1, 1250, 1000, 1000},
      {1, 1500, 1000, 1000},
      {1, 2000, 1000, 1000}
  };
  
  int iterations = 5;
  
  std::cout << "=========== FFT Benchmark Results ===========" << std::endl;
  std::cout << std::setw(20) << "Dimensions" << std::setw(20) << "CPU FFT (s)" 
            << std::setw(20) << "GPU FFT Host (s)" << std::setw(20) << "GPU FFT Device (s)" << std::endl;
  std::cout << "---------------------------------------------" << std::endl;
  
  for (const auto& config : configs) {
    int n4 = config[0];
    int n3 = config[1];
    int n2 = config[2];
    int n1 = config[3];
    
    std::string dimensions = std::to_string(n1) + "x" + std::to_string(n2) + 
                              "x" + std::to_string(n3) + "x" + std::to_string(n4);
    
    double cpu_time = benchmark_cpu_fft(n1, n2, n3, n4, iterations);
    
    std::cout << std::setw(20) << dimensions << std::setw(20) << std::fixed << std::setprecision(6) << cpu_time;
    
    double gpu_host_time = benchmark_gpu_fft_host(n1, n2, n3, n4, iterations);
    double gpu_device_time = benchmark_gpu_fft_device(n1, n2, n3, n4, iterations);
    
    std::cout << std::setw(20) << std::fixed << std::setprecision(6) << gpu_host_time
              << std::setw(20) << std::fixed << std::setprecision(6) << gpu_device_time;
    
    std::cout << std::endl;
  }
  
  std::cout << "==============================================" << std::endl;
  
  return 0;
}