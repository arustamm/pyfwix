#include <complex_vector.h>
#include <prop_kernels.cuh>
#include <cuComplex.h>
#include <math.h>
#include <KernelLauncher.cuh>
#include <KernelLauncher.cu>

// Combined linearized reflection forward operator
__global__ void drefl_forward(
  complex_vector* __restrict__ model,  
  complex_vector* __restrict__ data,
  const complex_vector* __restrict__ model_slow,
  const complex_vector* __restrict__ model_den) {

  int NX = model->n[0];
  int NY = model->n[1];
  int NW = model->n[2];

  cuFloatComplex* dslow = model->mat; // Slowness perturbation
  cuFloatComplex* dden = model->mat + 2 * NX * NY * NW; // Density perturbation
  
  int data_dims[] = {NW, NY, NX};           // 3D data array
  int slice_dims[] = {2, NW, NY, NX};       // Background model slices

  int ix0 = threadIdx.x + blockDim.x * blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y * blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z * blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int iw = iw0; iw < NW; iw += jw) {
    for (int iy = iy0; iy < NY; iy += jy) {
      for (int ix = ix0; ix < NX; ix += jx) {

        int slice_nd_ind[] = {0, iw, iy, ix};
        size_t slice_ind_0 = ND_TO_FLAT(slice_nd_ind, slice_dims);
        size_t slice_ind_1 = slice_ind_0 + NW * NX * NY;

        int data_nd_ind[] = {iw, iy, ix};
        size_t data_ind = ND_TO_FLAT(data_nd_ind, data_dims);

        // Compute coefficients on-the-fly
        cuFloatComplex sqrt_s0 = csqrtf(model_slow->mat[slice_ind_0]);
        cuFloatComplex sqrt_s1 = csqrtf(model_slow->mat[slice_ind_1]);
        cuFloatComplex rho0 = model_den->mat[slice_ind_0];
        cuFloatComplex rho1 = model_den->mat[slice_ind_1];

        // Denominator: (rho1*sqrt(s0) + rho0*sqrt(s1))
        cuFloatComplex denom = cuCaddf(
            cuCmulf(rho1, sqrt_s0),
            cuCmulf(rho0, sqrt_s1)
        );

        cuFloatComplex inv_denom = cuCdivf(make_cuFloatComplex(1.0f, 0.0f), denom);
        cuFloatComplex inv_denom_sq = cuCmulf(inv_denom, inv_denom);

        // Coefficients computed directly
        cuFloatComplex s1s2sq = cuCmulf(cuCmulf(rho0, rho1), inv_denom_sq);
        cuFloatComplex r1r2sq = cuCmulf(
            cuCmulf(make_cuFloatComplex(2.0f, 0.0f), cuCmulf(sqrt_s0, sqrt_s1)), 
            inv_denom_sq
        );

        cuFloatComplex result = make_cuFloatComplex(0.0f, 0.0f);

        // Slowness perturbation contribution
        cuFloatComplex c = cuCdivf(sqrt_s1, sqrt_s0);
        cuFloatComplex sqrt_c = csqrtf(c);
        cuFloatComplex inv_sqrt_c = cuCdivf(make_cuFloatComplex(1.0f, 0.0f), sqrt_c);

        cuFloatComplex a = cuCmulf(sqrt_c, s1s2sq);
        cuFloatComplex b = cuCmulf(inv_sqrt_c, s1s2sq);

        result = cuCaddf(result, cuCsubf(
            cuCmulf(a, dslow[slice_ind_0]),
            cuCmulf(b, dslow[slice_ind_1])
        ));

        // Density perturbation contribution
        a = cuCmulf(rho0, r1r2sq);
        b = cuCmulf(rho1, r1r2sq);

        result = cuCaddf(result, cuCsubf(
            cuCmulf(a, dden[slice_ind_1]),
            cuCmulf(b, dden[slice_ind_0])
        ));

        data->mat[data_ind] = cuCaddf(data->mat[data_ind], result);
      }
    }
  }
}

// Combined linearized reflection adjoint operator
__global__ void drefl_adjoint(
    complex_vector* __restrict__ model,
    complex_vector* __restrict__ data,
    const complex_vector* __restrict__ model_slow,
    const complex_vector* __restrict__ model_den) {

  int NX = model->n[0];
  int NY = model->n[1];
  int NW = model->n[2];

  cuFloatComplex* dslow = model->mat; // Slowness perturbation
  cuFloatComplex* dden = model->mat + 2 * NX * NY * NW; // Density perturbation
  
  int data_dims[] = {NW, NY, NX};           // 3D data array
  int slice_dims[] = {2, NW, NY, NX};       // Background model slices

  int ix0 = threadIdx.x + blockDim.x * blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y * blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z * blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int iw = iw0; iw < NW; iw += jw) {
    for (int iy = iy0; iy < NY; iy += jy) {
      for (int ix = ix0; ix < NX; ix += jx) {

        int slice_nd_ind[] = {0, iw, iy, ix};
        size_t slice_ind_0 = ND_TO_FLAT(slice_nd_ind, slice_dims);
        size_t slice_ind_1 = slice_ind_0 + NW * NX * NY;

        int data_nd_ind[] = {iw, iy, ix};
        size_t data_ind = ND_TO_FLAT(data_nd_ind, data_dims);

        cuFloatComplex data_val = data->mat[data_ind];

        // Compute coefficients on-the-fly (same as forward)
        cuFloatComplex sqrt_s0 = csqrtf(model_slow->mat[slice_ind_0]);
        cuFloatComplex sqrt_s1 = csqrtf(model_slow->mat[slice_ind_1]);
        cuFloatComplex rho0 = model_den->mat[slice_ind_0];
        cuFloatComplex rho1 = model_den->mat[slice_ind_1];

        cuFloatComplex denom = cuCaddf(
            cuCmulf(rho1, sqrt_s0),
            cuCmulf(rho0, sqrt_s1)
        );

        cuFloatComplex inv_denom = cuCdivf(make_cuFloatComplex(1.0f, 0.0f), denom);
        cuFloatComplex inv_denom_sq = cuCmulf(inv_denom, inv_denom);

        cuFloatComplex s1s2sq = cuCmulf(cuCmulf(rho0, rho1), inv_denom_sq);
        cuFloatComplex r1r2sq = cuCmulf(
            cuCmulf(make_cuFloatComplex(2.0f, 0.0f), cuCmulf(sqrt_s0, sqrt_s1)), 
            inv_denom_sq
        );

        // Slowness perturbation adjoint
        cuFloatComplex c = cuCdivf(sqrt_s1, sqrt_s0);
        cuFloatComplex sqrt_c = csqrtf(c);
        cuFloatComplex inv_sqrt_c = cuCdivf(make_cuFloatComplex(1.0f, 0.0f), sqrt_c);

        cuFloatComplex a = cuConjf(cuCmulf(sqrt_c, s1s2sq));
        cuFloatComplex b = cuConjf(cuCmulf(inv_sqrt_c, s1s2sq));

        cuFloatComplex contrib_slow_0 = cuCmulf(a, data_val);
        cuFloatComplex contrib_slow_1 = cuCmulf(cuCmulf(make_cuFloatComplex(-1.0f, 0.0f), b), data_val);

        atomicAdd(&(dslow[slice_ind_0].x), contrib_slow_0.x);
        atomicAdd(&(dslow[slice_ind_0].y), contrib_slow_0.y);
        atomicAdd(&(dslow[slice_ind_1].x), contrib_slow_1.x);
        atomicAdd(&(dslow[slice_ind_1].y), contrib_slow_1.y);

        // Density perturbation adjoint
        a = cuConjf(cuCmulf(rho0, r1r2sq));
        b = cuConjf(cuCmulf(rho1, r1r2sq));

        cuFloatComplex contrib_den_1 = cuCmulf(a, data_val);
        cuFloatComplex contrib_den_0 = cuCmulf(cuCmulf(make_cuFloatComplex(-1.0f, 0.0f), b), data_val);

        atomicAdd(&(dden[slice_ind_1].x), contrib_den_1.x);
        atomicAdd(&(dden[slice_ind_1].y), contrib_den_1.y);
        atomicAdd(&(dden[slice_ind_0].x), contrib_den_0.x);
        atomicAdd(&(dden[slice_ind_0].y), contrib_den_0.y);
      }
    }
  }
}

