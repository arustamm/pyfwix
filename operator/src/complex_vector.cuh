#pragma once
#include <complex_vector.h>

__global__ void add(complex_vector* vec1, complex_vector* vec2);
void launch_add(complex_vector* vec1, complex_vector* vec2, dim3 grid, dim3 block, cudaStream_t stream);
