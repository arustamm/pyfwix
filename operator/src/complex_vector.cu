#include <complex_vector.h>
#include <cuComplex.h>

__global__ void add(cuFloatComplex* vec1, cuFloatComplex* vec2, int N) {

  int i0 = threadIdx.x + blockDim.x*blockIdx.x;
  int j = blockDim.x * gridDim.x;

  for (int i=i0; i < N; i += j)
    vec1[i] = cuCaddf(vec1[i], vec2[i]);
};
void launch_add(complex_vector* vec1, complex_vector* vec2, dim3 grid, dim3 block, cudaStream_t stream) {
  add<<<grid, block, 0, stream>>>(vec1->mat, vec2->mat, vec1->nelem);
};

