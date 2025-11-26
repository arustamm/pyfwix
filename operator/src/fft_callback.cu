#include "complex_vector.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <iostream>

__device__ void CB_ortho(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr) {
  int* n = static_cast<int*>(callerInfo);
  float norm_factor = rsqrtf(static_cast<float>(n[0]));
  ((cufftComplex*)dataOut)[offset] = cuCmulf(element, make_cuFloatComplex(norm_factor, 0.0f));
}
__device__ cufftCallbackStoreC d_storeCallbackPtr = CB_ortho;

cufftCallbackStoreC get_host_callback_ptr() {
  cufftCallbackStoreC h_storeCallbackPtr;
  void * d_addr = nullptr;

  CHECK_CUDA_ERROR(cudaGetSymbolAddress(&d_addr, d_storeCallbackPtr));
  CHECK_CUDA_ERROR(cudaMemcpy(&h_storeCallbackPtr, d_addr, sizeof(h_storeCallbackPtr), cudaMemcpyDeviceToHost));
  
  return h_storeCallbackPtr;
}

