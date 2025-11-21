#include <complex_vector.h>
#include <prop_kernels.cuh>
#include <cuComplex.h>
#include <math.h> // for standard math functions like sqrt and fabs
#include <KernelLauncher.cuh>
#include <KernelLauncher.cu>

__global__ void refl_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
  const complex_vector* __restrict__ slow_slice, const complex_vector* __restrict__ den_slice) {

  int NX = slow_slice->n[0];
  int NY = slow_slice->n[1];
  int NW = model->n[2];
  int NS = model->n[3];
  int wfld_dims[] = {NS, NW, model->n[1], model->n[0]};
  int slice_dims[] = {2, NW, NY, NX};

  int ix0 = threadIdx.x + blockDim.x*blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y*blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z*blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int is=0; is < NS; ++is) {
  for (int iw=iw0; iw < NW; iw += jw) {
    for (int iy=iy0; iy < NY; iy += jy) {
      for (int ix=ix0; ix < NX; ix += jx) {

          int slice_nd_ind[] = {0, iw, iy, ix};
          size_t slice_ind_0 = ND_TO_FLAT(slice_nd_ind, slice_dims);
          size_t slice_ind_1 = slice_ind_0 + NW*NX*NY;

          int wfld_nd_ind[] = {is, iw, iy, ix};
          size_t wfld_ind = ND_TO_FLAT(wfld_nd_ind, wfld_dims);

          cuFloatComplex r = cuCdivf(
            cuCsubf(cuCmulf(den_slice->mat[slice_ind_1], csqrtf(slow_slice->mat[slice_ind_0])), cuCmulf(den_slice->mat[slice_ind_0], csqrtf(slow_slice->mat[slice_ind_1]))),
            cuCaddf(cuCmulf(den_slice->mat[slice_ind_1], csqrtf(slow_slice->mat[slice_ind_0])),cuCmulf(den_slice->mat[slice_ind_0],csqrtf(slow_slice->mat[slice_ind_1])))
          );

          data->mat[wfld_ind] = cuCaddf(data->mat[wfld_ind], cuCmulf(model->mat[wfld_ind], r));  
        }
      }
    }
  }
};

__global__ void refl_adjoint(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
  const complex_vector* __restrict__ slow_slice, const complex_vector* __restrict__ den_slice) {

  int NX = slow_slice->n[0];
  int NY = slow_slice->n[1];
  int NW = model->n[2];
  int NS = model->n[3];
  int wfld_dims[] = {NS, NW, model->n[1], model->n[0]};
  int slice_dims[] = {2, NW, NY, NX};

  int ix0 = threadIdx.x + blockDim.x*blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y*blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z*blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int is=0; is < NS; ++is) {
  for (int iw=iw0; iw < NW; iw += jw) {
    for (int iy=iy0; iy < NY; iy += jy) {
      for (int ix=ix0; ix < NX; ix += jx) {

          int slice_nd_ind[] = {0, iw, iy, ix};
          size_t slice_ind_0 = ND_TO_FLAT(slice_nd_ind, slice_dims);
          size_t slice_ind_1 = slice_ind_0 + NW*NX*NY;

          int wfld_nd_ind[] = {is, iw, iy, ix};
          size_t wfld_ind = ND_TO_FLAT(wfld_nd_ind, wfld_dims);

          cuFloatComplex r = cuCdivf(
            cuCsubf(cuCmulf(den_slice->mat[slice_ind_1], csqrtf(slow_slice->mat[slice_ind_0])), cuCmulf(den_slice->mat[slice_ind_0], csqrtf(slow_slice->mat[slice_ind_1]))),
            cuCaddf(cuCmulf(den_slice->mat[slice_ind_1], csqrtf(slow_slice->mat[slice_ind_0])),cuCmulf(den_slice->mat[slice_ind_0],csqrtf(slow_slice->mat[slice_ind_1])))
          );

          model->mat[wfld_ind] = cuCaddf(model->mat[wfld_ind], cuCmulf(data->mat[wfld_ind],cuConjf(r)));  
        }
      }
    }
  }
};

__global__ void refl_forward_in(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
  const complex_vector* __restrict__ slow_slice, const complex_vector* __restrict__ den_slice) {

  int NX = slow_slice->n[0];
  int NY = slow_slice->n[1];
  int NW = model->n[2];
  int NS = model->n[3];
  int wfld_dims[] = {NS, NW, model->n[1], model->n[0]};
  int slice_dims[] = {NW, NY, NX};

  int ix0 = threadIdx.x + blockDim.x*blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y*blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z*blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int is=0; is < NS; ++is) {
  for (int iw=iw0; iw < NW; iw += jw) {
    for (int iy=iy0; iy < NY; iy += jy) {
      for (int ix=ix0; ix < NX; ix += jx) {

          int slice_nd_ind[] = {iw, iy, ix};
          size_t slice_ind_0 = ND_TO_FLAT(slice_nd_ind, slice_dims);
          size_t slice_ind_1 = slice_ind_0 + NW*NX*NY;

          int wfld_nd_ind[] = {is, iw, iy, ix};
          size_t wfld_ind = ND_TO_FLAT(wfld_nd_ind, wfld_dims);

          cuFloatComplex r = cuCdivf(
            cuCsubf(cuCmulf(den_slice->mat[slice_ind_1], csqrtf(slow_slice->mat[slice_ind_0])), cuCmulf(den_slice->mat[slice_ind_0], csqrtf(slow_slice->mat[slice_ind_1]))),
            cuCaddf(cuCmulf(den_slice->mat[slice_ind_1], csqrtf(slow_slice->mat[slice_ind_0])),cuCmulf(den_slice->mat[slice_ind_0],csqrtf(slow_slice->mat[slice_ind_1])))
          );
          
          data->mat[wfld_ind] = cuCmulf(model->mat[wfld_ind], r);  
        }
      }
    }
  }
};

__global__ void refl_adjoint_in(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
  const complex_vector* __restrict__ slow_slice,  const complex_vector* __restrict__ den_slice) {

  int NX = slow_slice->n[0];
  int NY = slow_slice->n[1];
  int NW = data->n[2];
  int NS = data->n[3];
  int wfld_dims[] = {NS, NW, data->n[1], data->n[0]};
  int slice_dims[] = {NW, NY, NX};

  int ix0 = threadIdx.x + blockDim.x*blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y*blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z*blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int is=0; is < NS; ++is) {
  for (int iw=iw0; iw < NW; iw += jw) {
    for (int iy=iy0; iy < NY; iy += jy) {
      for (int ix=ix0; ix < NX; ix += jx) {

          int slice_nd_ind[] = {iw, iy, ix};
          size_t slice_ind_0 = ND_TO_FLAT(slice_nd_ind, slice_dims);
          size_t slice_ind_1 = slice_ind_0 + NW*NX*NY;

          int wfld_nd_ind[] = {is, iw, iy, ix};
          size_t wfld_ind = ND_TO_FLAT(wfld_nd_ind, wfld_dims);

          cuFloatComplex r = cuCdivf(
            cuCsubf(cuCmulf(den_slice->mat[slice_ind_1], csqrtf(slow_slice->mat[slice_ind_0])), cuCmulf(den_slice->mat[slice_ind_0], csqrtf(slow_slice->mat[slice_ind_1]))),
            cuCaddf(cuCmulf(den_slice->mat[slice_ind_1], csqrtf(slow_slice->mat[slice_ind_0])),cuCmulf(den_slice->mat[slice_ind_0],csqrtf(slow_slice->mat[slice_ind_1])))
          );

          model->mat[wfld_ind] = cuCmulf(data->mat[wfld_ind],cuConjf(r));  
        }
      }
    }
  }
};
