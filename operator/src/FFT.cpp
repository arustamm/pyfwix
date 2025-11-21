#include "FFT.h"

using namespace SEP;

cuFFT2d::cuFFT2d(const std::shared_ptr<hypercube>& domain, complex_vector* model, complex_vector* data, 
dim3 grid, dim3 block, cudaStream_t stream)
: CudaOperator<complex4DReg, complex4DReg>(domain, domain, model, data, grid, block, stream) {
  // create plan  

  _grid_ = {32, 4, 4};
  _block_ = {16, 16, 4};
  
  NX = getDomain()->getAxis(1).n;
  NY = getDomain()->getAxis(2).n;
  SIZE = NX*NY;
  BATCH = getDomain()->getN123() / SIZE;

  // Allocate device memory for size
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_SIZE, sizeof(int)));
  CHECK_CUDA_ERROR(cudaMemcpy(d_SIZE, &SIZE, sizeof(int), cudaMemcpyHostToDevice));

  int rank = 2;
  int dims[2] = {NY, NX};

  cufftPlanMany(&plan, rank, dims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, BATCH);
  // set the callback to make it orthogonal
  auto h_storeCallbackPtr = get_host_callback_ptr();
  cufftXtSetCallback(plan, (void **)&h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX, (void **)&d_SIZE);

  temp = make_complex_vector(domain, _grid_, _block_, stream);

  cufftSetStream(plan, stream);
};

// this is on-device function
void cuFFT2d::cu_forward(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
  if (!add) data->zero();
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(model, sizeof(complex_vector), cudaCpuDeviceId, _stream_)); 
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(data, sizeof(complex_vector), cudaCpuDeviceId, _stream_)); 
  cufftExecC2C(plan, model->mat, temp->mat, CUFFT_FORWARD);
  // CHECK_CUDA_ERROR(cudaStreamSynchronize(_stream_));
  data->add(temp);
};

// this is on-device function
void cuFFT2d::cu_adjoint(bool add, complex_vector* __restrict__ model, complex_vector* __restrict__ data) {
  if (!add) model->zero();
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(model, sizeof(complex_vector), cudaCpuDeviceId, _stream_)); 
  CHECK_CUDA_ERROR(cudaMemPrefetchAsync(data, sizeof(complex_vector), cudaCpuDeviceId, _stream_)); 
  cufftExecC2C(plan, data->mat, temp->mat, CUFFT_INVERSE);
  // CHECK_CUDA_ERROR(cudaStreamSynchronize(_stream_));
  model->add(temp);
};

// this is on-device function
void cuFFT2d::cu_forward(__restrict__ complex_vector* data) {
  cufftExecC2C(plan, data->mat, data->mat, CUFFT_FORWARD);
  // CHECK_CUDA_ERROR(cudaStreamSynchronize(_stream_));
};

// this is on-device function
void cuFFT2d::cu_adjoint(__restrict__ complex_vector* data) {
  cufftExecC2C(plan, data->mat, data->mat, CUFFT_INVERSE);
  // CHECK_CUDA_ERROR(cudaStreamSynchronize(_stream_));
};

