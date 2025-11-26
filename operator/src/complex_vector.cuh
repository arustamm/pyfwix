#pragma once
#include <complex_vector.h>

__global__ void add(cuFloatComplex* vec1, cuFloatComplex* vec2, int N);
__global__ void scale(cuFloatComplex* vec1, int N, float scale);
void launch_add(complex_vector* vec1, complex_vector* vec2, dim3 grid, dim3 block, cudaStream_t stream);
void launch_scale(complex_vector* vec1, float scale, dim3 grid, dim3 block, cudaStream_t stream);
