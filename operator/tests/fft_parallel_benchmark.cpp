#include <complex4DReg.h>
#include "fftw3.h"
#include <FFT.h>
#include <benchmark/benchmark.h>
#include <complex_vector.h>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>

using namespace SEP;

class FFT2d {
public:
  FFT2d(const std::shared_ptr<hypercube>& domain, const std::shared_ptr<hypercube>& range) {
    NX = domain->getAxis(1).n;
    NY = domain->getAxis(2).n;
    BATCH = domain->getN123() / (NX*NY);
    SIZE = domain->getN123();  
    int dims[2] = {NX, NY};
		// similar to cumalloc in cufft
    fftwf_init_threads();
    int nthreads = std::thread::hardware_concurrency();
    fftwf_plan_with_nthreads(nthreads);
    _model = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*SIZE);
    _data = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*SIZE);

    _fwd_plan = fftwf_plan_many_dft(2, dims, BATCH,
	                             _model, NULL, 1, 0,
	                             _data, NULL, 1, 0,
	                             FFTW_FORWARD,FFTW_PATIENT  );
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

class cpuFFTBenchmark : public benchmark::Fixture {
 protected:
  void SetUp(::benchmark::State& state) override {
    int n1 = state.range(3);
    int n2 = state.range(2);
    int n3 = state.range(1);
    int n4 = state.range(0);
    auto hyper = std::make_shared<hypercube>(n1, n2, n3, n4);
    model = std::make_shared<complex4DReg>(hyper);
    data = std::make_shared<complex4DReg>(hyper);
    model->set(1.f);
    FFT = std::make_unique<FFT2d>(hyper, hyper);
  }
  std::unique_ptr<FFT2d> FFT;
  std::shared_ptr<complex4DReg> model;
  std::shared_ptr<complex4DReg> data;
  int n1, n2, n3, n4;
};

BENCHMARK_DEFINE_F(cpuFFTBenchmark, forward_host)(benchmark::State& state){
  for (auto _ : state){
    auto start = std::chrono::high_resolution_clock::now();
    FFT->forward(false, model, data);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
    
};
BENCHMARK_REGISTER_F(cpuFFTBenchmark, forward_host)
-> Args({1, 1, 100, 100}) 
-> Args({1, 5, 100, 100}) 
-> Args({1, 10, 100, 100}) 

-> Args({1, 1, 100, 500}) 
-> Args({1, 5, 100, 500}) 
-> Args({1, 10, 100, 500}) 

-> Args({1, 1, 500, 5000}) 
-> Args({1, 5, 500, 5000}) 
-> Args({1, 10, 500, 5000}) 

-> Args({1, 1, 1000, 1000}) 
-> Args({1, 5, 1000, 1000}) 
-> Args({1, 10, 1000, 1000}) 

-> Iterations(10)
->UseManualTime();


class FFTBenchmark : public benchmark::Fixture {
 protected:
  void SetUp(::benchmark::State& state) override {
    int n1 = state.range(3);
    int n2 = state.range(2);
    int n3 = state.range(1);
    int n4 = state.range(0);
    auto hyper = std::make_shared<hypercube>(n1, n2, n3, n4);
    model = std::make_shared<complex4DReg>(hyper);
    data = std::make_shared<complex4DReg>(hyper);
    model->set(1.f);
    cuFFT = std::make_unique<cuFFT2d>(hyper);
  }
  std::unique_ptr<cuFFT2d> cuFFT;
  std::shared_ptr<complex4DReg> model;
  std::shared_ptr<complex4DReg> data;
  int n1, n2, n3, n4;
};

BENCHMARK_DEFINE_F(FFTBenchmark, forward_device)(benchmark::State& state){
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    cuFFT->cu_forward(false, cuFFT->model_vec, cuFFT->data_vec);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
   
};
BENCHMARK_REGISTER_F(FFTBenchmark, forward_device)
-> Args({1, 1, 100, 100}) 
-> Args({1, 5, 100, 100}) 
-> Args({1, 10, 100, 100}) 

-> Args({1, 1, 100, 500}) 
-> Args({1, 5, 100, 500}) 
-> Args({1, 10, 100, 500}) 

-> Args({1, 1, 500, 5000}) 
-> Args({1, 5, 500, 5000}) 
-> Args({1, 10, 500, 5000}) 

-> Args({1, 1, 1000, 1000}) 
-> Args({1, 5, 1000, 1000}) 
-> Args({1, 10, 1000, 1000}) 

-> Iterations(10)
-> UseManualTime();

BENCHMARK_MAIN();