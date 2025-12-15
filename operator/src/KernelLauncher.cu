#include <KernelLauncher.cuh>

template <typename... Args>
KernelLauncher<Args...>::KernelLauncher() 
    : _grid_(1), _block_(1), _fwd_kernel_(nullptr), _adj_kernel_(nullptr), _stream_(0) {}

template <typename... Args>
KernelLauncher<Args...>::KernelLauncher(
    FwdKernel fwd_kernel, AdjKernel adj_kernel, 
    dim3 grid, dim3 block, cudaStream_t stream) 
    : _grid_(grid), _block_(block), _fwd_kernel_(fwd_kernel), 
      _adj_kernel_(adj_kernel), _stream_(stream) {}

template <typename... Args>
KernelLauncher<Args...>::KernelLauncher(
    FwdKernel fwd_kernel, dim3 grid, dim3 block, cudaStream_t stream)
    : _grid_(grid), _block_(block), _fwd_kernel_(fwd_kernel), 
      _adj_kernel_(nullptr), _stream_(stream) {}

template <typename... Args>
KernelLauncher<Args...>::~KernelLauncher() {}

template <typename... Args>
void KernelLauncher<Args...>::run_fwd(complex_vector* __restrict__ model, complex_vector* __restrict__ data, Args... args) {
    _fwd_kernel_<<<_grid_, _block_, 0, _stream_>>>(model, data, args...);
    CHECK_CUDA_ERROR( cudaPeekAtLastError() );
  };

template <typename... Args>
void KernelLauncher<Args...>::run_adj(complex_vector* __restrict__ model, complex_vector* __restrict__ data, Args... args) {
    _adj_kernel_<<<_grid_, _block_, 0, _stream_>>>(model, data, args...);
    CHECK_CUDA_ERROR( cudaPeekAtLastError() );
  };
