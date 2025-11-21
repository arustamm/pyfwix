#include <complex4DReg.h>
#include <benchmark/benchmark.h>
#include <complex_vector.h>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>

#include  <RefSampler.h>
#include <OneStep.h>
#include <jsonParamObj.h>

using namespace SEP;

class PSPIBenchmark : public benchmark::Fixture {
 protected:
  void SetUp(::benchmark::State& state) override {
    int nx = state.range(4);
    int ny = state.range(3);
    int nw = state.range(2);
    int ns = state.range(1);
    int nref = state.range(0);

    int nz = 10;
    auto hyper = std::make_shared<hypercube>(nx, ny, nw, ns);
    model = std::make_shared<complex4DReg>(hyper);
    data = std::make_shared<complex4DReg>(hyper);

    auto slow4d = std::make_shared<complex4DReg>(nx, ny, nw, nz);
    slow4d->random();

    Json::Value root;
    root["nref"] = nref;
    auto par = std::make_shared<jsonParamObj>(root);

    pspi = std::make_unique<PSPI>(hyper, slow4d, par);
    pspi->set_depth(5);
  }
  std::unique_ptr<PSPI> pspi;
  std::shared_ptr<complex4DReg> model;
  std::shared_ptr<complex4DReg> data;
};


BENCHMARK_DEFINE_F(PSPIBenchmark, forward_device)(benchmark::State& state){
  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    pspi->cu_forward(false, pspi->model_vec, pspi->data_vec);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
   
};
BENCHMARK_REGISTER_F(PSPIBenchmark, forward_device)
-> Args({1, 1, 1, 1000, 1000}) 
-> Args({1, 1, 10, 1000, 1000}) 
-> Args({1, 1, 100, 1000, 1000}) 

// -> Args({5, 1, 1, 1000, 1000}) 
// -> Args({5, 1, 10, 1000, 1000}) 
// -> Args({5, 1, 100, 1000, 1000}) 

// -> Args({10, 1, 1, 1000, 1000}) 
// -> Args({10, 1, 10, 1000, 1000}) 
// -> Args({10, 1, 100, 1000, 1000}) 

// -> Args({20, 1, 1, 1000, 1000}) 
// -> Args({20, 1, 10, 1000, 1000}) 
// -> Args({20, 1, 100, 1000, 1000}) 

// -> Args({50, 1, 1, 1000, 1000}) 
// -> Args({50, 1, 10, 1000, 1000}) 
// -> Args({50, 1, 100, 1000, 1000}) 

-> Iterations(2)
-> Threads(1)
-> UseManualTime();

BENCHMARK_MAIN();