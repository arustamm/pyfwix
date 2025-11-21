#include <complex4DReg.h>
#include "fftw3.h"
#include <FFT.h>
#include <benchmark/benchmark.h>
#include <complex_vector.h>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>

using namespace SEP;

class FFTBenchmark : public benchmark::Fixture
{
protected:
  void SetUp(::benchmark::State &state) override
  {
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

BENCHMARK_DEFINE_F(FFTBenchmark, forward_device)
(benchmark::State &state)
{
  for (auto _ : state)
  {
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
    ->ArgsProduct({{1,5,10,50,100},{1,5,10,50,100}, {1000}, {1000}})
    ->Iterations(2)
    ->UseManualTime();

BENCHMARK_MAIN();