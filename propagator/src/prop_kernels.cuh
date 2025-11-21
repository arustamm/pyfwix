#pragma once
#include <cuda_runtime.h>
#include <complex_vector.h>
#include <cuComplex.h>
#include <KernelLauncher.cuh>

// Explicit template instantiations for all used combinations
template class KernelLauncher<const float*, const float*, const float*, const cuFloatComplex*, float, float>;
template class KernelLauncher<int, const int*>;
template class KernelLauncher<const float*, const float*, const float*, const int*, float, float, int>;
template class KernelLauncher<const complex_vector*, const complex_vector*>;
template class KernelLauncher<int, int>;
template class KernelLauncher<const float*, const float*, const float*, int>;
template class KernelLauncher<const complex_vector*, float, int, float>;
template class KernelLauncher<const float*, float>;
template class KernelLauncher<>; 
template class KernelLauncher<const complex_vector*>;

// phase shift
__global__ void ps_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
  const float* __restrict__ w2, const float* __restrict__ kx, const float* __restrict__ ky, const cuFloatComplex* __restrict__ slow_ref, float dz, float eps);
__global__ void ps_adjoint(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
  const float* __restrict__ w2, const float* __restrict__ kx, const float* __restrict__ ky, const cuFloatComplex* __restrict__ slow_ref, float dz, float eps);
  __global__ void ps_inverse(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
    const float* __restrict__ w2, const float* __restrict__ kx, const float* __restrict__ ky, const cuFloatComplex* __restrict__ slow_ref, float dz, float eps);
typedef KernelLauncher<const float*, const float*, const float*, const cuFloatComplex*, float, float> PS_launcher;

// selector
__global__ void select_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, int value, const int* __restrict__ labels);
typedef KernelLauncher<int, const int*> Selector_launcher;

  // injection
__global__ void inj_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, const float* __restrict__ cx, const float* __restrict__ cy, const float* __restrict__ cz, const int* __restrict__ ids, float oz, float dz, int iz);
__global__ void inj_adjoint(complex_vector* __restrict__ model, complex_vector* __restrict__ data, const float* __restrict__ cx, const float* __restrict__ cy, const float* __restrict__ cz, const int* __restrict__ ids, float oz, float dz, int iz);
typedef KernelLauncher<const float*, const float*, const float*, const int*, float, float, int> Injection_launcher;
  
// reflection
  __global__ void refl_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, const complex_vector* slow_slice, const complex_vector* __restrict__ den_slice);
  __global__ void refl_adjoint(complex_vector* __restrict__ model, complex_vector* __restrict__ data, const complex_vector* __restrict__ slow_slice, const complex_vector* __restrict__ den_slice);
  __global__ void refl_forward_in(complex_vector* __restrict__ model, complex_vector* __restrict__ data, const complex_vector* __restrict__ slow_slice, const complex_vector* __restrict__ den_slice);
  __global__ void refl_adjoint_in(complex_vector* __restrict__ model, complex_vector* __restrict__ data, const complex_vector* __restrict__ slow_slice, const complex_vector* __restrict__ den_slice);
  typedef KernelLauncher<const complex_vector*, const complex_vector*> Refl_launcher;

// taper
__global__ void taper_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, int tapx, int tapy);
typedef KernelLauncher<int, int> Taper_launcher;

// scatter
__global__ void mult_kxky(complex_vector* __restrict__ model, complex_vector* __restrict__ data,
  const float* __restrict__ w, const float* __restrict__ kx, const float* __restrict__ ky, int it);
typedef KernelLauncher<const float*, const float*, const float*, int> Mult_kxky;

__global__ void slow_scale_fwd(complex_vector* __restrict__ model, complex_vector* __restrict__ data,
  const complex_vector* __restrict__ slow_slice, float coef, 
  int it, float eps);
__global__ void slow_scale_adj(complex_vector* __restrict__ model, complex_vector* __restrict__ data,
  const complex_vector* __restrict__ slow_slice, float coef, 
  int it, float eps);
typedef KernelLauncher<const complex_vector*, float, int, float> Slow_scale;

__global__ void scale_by_iw_fwd(complex_vector* __restrict__ model, complex_vector* __restrict__ data,
  const float* __restrict__ w, float dz);
__global__ void scale_by_iw_adj(complex_vector* __restrict__ model, complex_vector* __restrict__ data,
  const float* __restrict__ w, float dz);
typedef KernelLauncher<const float*, float> Scale_by_iw;

__global__ void pad_fwd(complex_vector* __restrict__ model, complex_vector* __restrict__ data);
__global__ void pad_adj(complex_vector* __restrict__ model, complex_vector* __restrict__ data);
typedef KernelLauncher<> Pad_launcher;

// imaging condition
__global__ void ic_fwd(complex_vector* __restrict__ model, complex_vector* __restrict__ data, const complex_vector* __restrict__ bg_wfld);
__global__ void ic_adj(complex_vector* __restrict__ model, complex_vector* __restrict__ data, const complex_vector* __restrict__ bg_wfld);
typedef KernelLauncher<const complex_vector*> IC_launcher;

// dreflect
__global__ void drefl_forward(
    complex_vector* __restrict__ model,
    complex_vector* __restrict__ data,
    const complex_vector* __restrict__ model_slow,
    const complex_vector* __restrict__ model_den);
__global__ void drefl_adjoint(
    complex_vector* __restrict__ model,
    complex_vector* __restrict__ data,
    const complex_vector* __restrict__ model_slow,
    const complex_vector* __restrict__ model_den);


__device__ __forceinline__ cuFloatComplex csqrtf(cuFloatComplex z) {
  float a = z.x;
  float b = z.y;

  if (a == 0.0f && b == 0.0f) {
      return make_cuFloatComplex(0.0f, 0.0f); // Handle zero case
  }

  float r = sqrtf(a * a + b * b);
  float x = sqrtf(0.5f * (r + a));
  float y = copysignf(sqrtf(0.5f * (r - a)), b); // Correctly handle sign of imaginary part

  return make_cuFloatComplex(x, y);
}
