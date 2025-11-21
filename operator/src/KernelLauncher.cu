#include <KernelLauncher.cuh>

template <typename... Args>
KernelLauncher<Args...>::KernelLauncher() {};

template <typename... Args>
KernelLauncher<Args...>::KernelLauncher(
void (*fwd_kernel)(complex_vector* __restrict__, complex_vector* __restrict__, Args...), 
void (*adj_kernel)(complex_vector* __restrict__, complex_vector* __restrict__, Args...), 
dim3 grid, dim3 block, cudaStream_t stream) : 
_grid_(grid), _block_(block), _fwd_kernel_(fwd_kernel), _adj_kernel_(adj_kernel), _stream_(stream) {};

template <typename... Args>
KernelLauncher<Args...>::KernelLauncher(
void (*fwd_kernel)(complex_vector* __restrict__, complex_vector* __restrict__, Args...), 
dim3 grid, dim3 block, cudaStream_t stream) :
 _grid_(grid), _block_(block), _fwd_kernel_(fwd_kernel), _stream_(stream) {};

template <typename... Args>
KernelLauncher<Args...>::~KernelLauncher() {};

template <typename... Args>
void KernelLauncher<Args...>::run_fwd(complex_vector* __restrict__ model, complex_vector* __restrict__ data, Args... args) {
    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(model, sizeof(complex_vector), cudaCpuDeviceId, _stream_)); 
    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(data, sizeof(complex_vector), cudaCpuDeviceId, _stream_)); 
    _fwd_kernel_<<<_grid_, _block_, 0, _stream_>>>(model, data, args...);
    CHECK_CUDA_ERROR( cudaPeekAtLastError() );
    // CHECK_CUDA_ERROR(cudaStreamSynchronize(_stream_));
  };

template <typename... Args>
void KernelLauncher<Args...>::run_adj(complex_vector* __restrict__ model, complex_vector* __restrict__ data, Args... args) {
    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(model, sizeof(complex_vector), cudaCpuDeviceId, _stream_)); 
    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(data, sizeof(complex_vector), cudaCpuDeviceId, _stream_)); 
    _adj_kernel_<<<_grid_, _block_, 0, _stream_>>>(model, data, args...);
    CHECK_CUDA_ERROR( cudaPeekAtLastError() );
    // CHECK_CUDA_ERROR(cudaStreamSynchronize(_stream_));
  };