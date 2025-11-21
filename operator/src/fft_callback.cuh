#pragma once

__device__ void CB_ortho(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr);
cufftCallbackStoreC get_host_callback_ptr();