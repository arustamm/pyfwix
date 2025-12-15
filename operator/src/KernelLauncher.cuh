#pragma once
#include <functional>
#include <tuple>
#include <complex_vector.h>
#include <cuda.h>
#include <iostream>
#include <map>

template <typename... Args>
class KernelLauncher {
public:
  using FwdKernel = void (*)(complex_vector* __restrict__, complex_vector* __restrict__, Args...);
  using AdjKernel = void (*)(complex_vector* __restrict__, complex_vector* __restrict__, Args...);


  KernelLauncher();
  KernelLauncher(FwdKernel fwd_kernel, AdjKernel adj_kernel, 
                  dim3 grid, dim3 block, cudaStream_t stream);
  
  KernelLauncher(FwdKernel fwd_kernel, 
                  dim3 grid, dim3 block, cudaStream_t stream);
  
  ~KernelLauncher();

  void run_fwd(complex_vector* __restrict__ model, complex_vector* __restrict__ data, Args... args);
  void run_adj(complex_vector* __restrict__ model, complex_vector* __restrict__ data, Args... args);

  void set_grid_block(dim3 grid, dim3 block) {
    _grid_ = grid;
    _block_ = block;
  }
  
private:
  dim3 _grid_, _block_;
  cudaStream_t _stream_;
  FwdKernel _fwd_kernel_;
  AdjKernel _adj_kernel_;

};
