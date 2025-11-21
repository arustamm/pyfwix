#include <complex_vector.h>
#include <prop_kernels.cuh>
#include <cuComplex.h>
#include <KernelLauncher.cuh>
#include <KernelLauncher.cu>

__global__ void ps_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
  const float* __restrict__  w2, const float* __restrict__  kx, const float* __restrict__  ky, const cuFloatComplex* __restrict__ slow_ref, float dz, float eps) {

  float a, b, c, re, im;
  size_t flat_ind;
  int NX = model->n[0];
  int NY = model->n[1];
  int NW = model->n[2];
  int NS = model->n[3];
  int dims[] = {NS, NW, NY, NX};

  int ix0 = threadIdx.x + blockDim.x*blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y*blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z*blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int iw=iw0; iw < NW; iw += jw) {
    float sre = cuCrealf(slow_ref[iw]);
    float sim = cuCimagf(slow_ref[iw]);
    for (int iy=iy0; iy < NY; iy += jy) {
      for (int ix=ix0; ix < NX; ix += jx) {
        a = w2[iw]*sre - (kx[ix]*kx[ix] + ky[iy]*ky[iy]);
        b = w2[iw]*(sim-eps*sre);
        c = sqrtf(a*a + b*b);

        re = sqrtf((c+a)/2);
        im = -sqrtf((c-a)/2);

        float att = exp(im*dz);
        float coss;
        float sinn;
        sincos(re*dz, &sinn, &coss);

        for (int is=0; is < NS; is++) {

          int nd_ind[] = {is, iw, iy, ix};
          flat_ind = ND_TO_FLAT(nd_ind, dims);

          float mre = cuCrealf(model->mat[flat_ind]);
          float mim = cuCimagf(model->mat[flat_ind]);

          re = att * (mre * coss + mim * sinn);
          im = att * (-mre * sinn + mim * coss);

          data->mat[flat_ind] = cuCaddf(data->mat[flat_ind], make_cuFloatComplex(re, im)); 
        }
      }
    }
  }
};

// __global__ void ps_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
//   const float* __restrict__ w2, const float* __restrict__ kx, const float* __restrict__ ky, 
//   const cuFloatComplex* __restrict__ slow_ref, float dz, float eps) {

//   const int NX = model->n[0];
//   const int NY = model->n[1];
//   const int NW = model->n[2];
//   const int NS = model->n[3];
//   const int dims[] = {NS, NW, NY, NX};
  
//   // Calculate linear thread ID in the grid
//   const int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   const int total_threads = gridDim.x * blockDim.x;
  
//   // Each thread processes multiple elements across the entire 4D domain
//   const int total_elements = NX * NY * NW * NS;
  
//   // Process elements with stride equal to total number of threads
//   for (int idx = tid; idx < total_elements; idx += total_threads) {
//     // Convert linear index to 4D coordinates
//     int ix = idx % NX;
//     int iy = (idx / NX) % NY;
//     int iw = (idx / (NX * NY)) % NW;
//     int is = idx / (NX * NY * NW);
    
//     // Your original computation
//     float sre = cuCrealf(slow_ref[iw]);
//     float sim = cuCimagf(slow_ref[iw]);
    
//     float a = w2[iw]*sre - (kx[ix]*kx[ix] + ky[iy]*ky[iy]);
//     float b = w2[iw]*(sim-eps*sre);
//     float c = sqrtf(a*a + b*b);
    
//     float re, im;
//     if (b <= 0) {
//       re = sqrtf((c+a)/2);
//       im = -sqrtf((c-a)/2);
//     }
//     else {
//       re = -sqrtf((c+a)/2);
//       im = -sqrtf((c-a)/2);
//     }
    
//     // Existing indexing method
//     // int nd_ind[] = {is, iw, iy, ix};
//     // size_t flat_ind = ND_TO_FLAT(nd_ind, dims);
    
//     float att = exp(im*dz);
//     float coss = cos(re*dz);
//     float sinn = sin(re*dz);
    
//     float mre = cuCrealf(model->mat[idx]);
//     float mim = cuCimagf(model->mat[idx]);
    
//     re = att * (mre * coss + mim * sinn);
//     im = att * (-mre * sinn + mim * coss);
    
//     data->mat[idx] = cuCaddf(data->mat[idx], make_cuFloatComplex(re, im));
//   }
// }


__global__ void ps_adjoint(complex_vector* __restrict__ model, complex_vector* __restrict__ data, const float* __restrict__ w2, const float* __restrict__ kx, const float* __restrict__ ky, const cuFloatComplex* __restrict__ slow_ref, float dz, float eps) {
  
  float a, b, c, re, im;
  size_t flat_ind;
  int NX = model->n[0];
  int NY = model->n[1];
  int NW = model->n[2];
  int NS = model->n[3];
  int dims[] = {NS, NW, NY, NX};

  int ix0 = threadIdx.x + blockDim.x*blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y*blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z*blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int iw=iw0; iw < NW; iw += jw) {
    float sre = cuCrealf(slow_ref[iw]);
    float sim = cuCimagf(slow_ref[iw]);
    for (int iy=iy0; iy < NY; iy += jy) {
      for (int ix=ix0; ix < NX; ix += jx) {
        a = w2[iw]*sre - (kx[ix]*kx[ix] + ky[iy]*ky[iy]);
        b = w2[iw]*(sim-eps*sre);
        c = sqrtf(a*a + b*b);
        if (b <= 0) {
          re = sqrtf((c+a)/2);
          im = -sqrtf((c-a)/2);
        }
        else {
          re = -sqrtf((c+a)/2);
				  im = -sqrtf((c-a)/2);
        }

        float att = exp(im*dz);
        float coss;
        float sinn;
        sincos(re*dz, &sinn, &coss);
        
        for (int is=0; is < NS; is++) {
          // convert 4d index to flat index
          int nd_ind[] = {is, iw, iy, ix};
          flat_ind = ND_TO_FLAT(nd_ind, dims);

          float dre = cuCrealf(data->mat[flat_ind]);
          float dim = cuCimagf(data->mat[flat_ind]);

          re = att * (dre * coss - dim * sinn);
          im = att * (dre * sinn + dim * coss);

          model->mat[flat_ind] = cuCaddf(model->mat[flat_ind], make_cuFloatComplex(re, im));
        }
      }
    }
  }
};

// __global__ void ps_adjoint(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
//   const float* __restrict__ w2, const float* __restrict__ kx, const float* __restrict__ ky, 
//   const cuFloatComplex* __restrict__ slow_ref, float dz, float eps) {

//   const int NX = model->n[0];
//   const int NY = model->n[1];
//   const int NW = model->n[2];
//   const int NS = model->n[3];
//   const int dims[] = {NS, NW, NY, NX};
  
//   // Calculate linear thread ID in the grid
//   const int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   const int total_threads = gridDim.x * blockDim.x;
  
//   // Each thread processes multiple elements across the entire 4D domain
//   const int total_elements = NX * NY * NW * NS;
  
//   // Process elements with stride equal to total number of threads
//   for (int idx = tid; idx < total_elements; idx += total_threads) {
//     // Convert linear index to 4D coordinates
//     int ix = idx % NX;
//     int iy = (idx / NX) % NY;
//     int iw = (idx / (NX * NY)) % NW;
//     int is = idx / (NX * NY * NW);
    
//     // Your original computation
//     float sre = cuCrealf(slow_ref[iw]);
//     float sim = cuCimagf(slow_ref[iw]);
    
//     float a = w2[iw]*sre - (kx[ix]*kx[ix] + ky[iy]*ky[iy]);
//     float b = w2[iw]*(sim-eps*sre);
//     float c = sqrtf(a*a + b*b);
    
//     float re, im;
//     if (b <= 0) {
//       re = sqrtf((c+a)/2);
//       im = -sqrtf((c-a)/2);
//     }
//     else {
//       re = -sqrtf((c+a)/2);
//       im = -sqrtf((c-a)/2);
//     }
    
//     // Existing indexing method
//     // int nd_ind[] = {is, iw, iy, ix};
//     // size_t flat_ind = ND_TO_FLAT(nd_ind, dims);
    
//     float att = exp(im*dz);
//     float coss = cos(re*dz);
//     float sinn = sin(re*dz);
    
//     float dre = cuCrealf(data->mat[idx]);
//     float dim = cuCimagf(data->mat[idx]);
    
//     re = att * (dre * coss - dim * sinn);
//     im = att * (dre * sinn + dim * coss);
    
//     model->mat[idx] = cuCaddf(model->mat[idx], make_cuFloatComplex(re, im));
//   }
// }


__global__ void ps_inverse(complex_vector* __restrict__ model, complex_vector* __restrict__ data, const float* w2, const float* kx, const float* ky, const cuFloatComplex* slow_ref, float dz, float eps) {
  
  float a, b, c, re, im;
  size_t flat_ind;

  int NX = model->n[0];
  int NY = model->n[1];
  int NW = model->n[2];
  int NS = model->n[3];
  int dims[] = {NS, NW, NY, NX};

  int ix0 = threadIdx.x + blockDim.x*blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y*blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z*blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int is=0; is < NS; ++is) {

  for (int iw=iw0; iw < NW; iw += jw) {
    float sre = cuCrealf(slow_ref[iw]);
    float sim = cuCimagf(slow_ref[iw]);
    for (int iy=iy0; iy < NY; iy += jy) {
      for (int ix=ix0; ix < NX; ix += jx) {
        a = w2[iw]*sre - (kx[ix]*kx[ix] + ky[iy]*ky[iy]);
        b = w2[iw]*(sim-eps*sre);
        c = sqrtf(a*a + b*b);
        if (b <= 0) {
          re = sqrtf((c+a)/2);
          im = sqrtf((c-a)/2);
        }
        else {
          re = -sqrt((c+a)/2);
				  im = sqrt((c-a)/2);
        }

          // convert 4d index to flat index
          int nd_ind[] = {is, iw, iy, ix};
          flat_ind = ND_TO_FLAT(nd_ind, dims);

          float att = exp(im*dz);
          float coss = cos(re*dz);
          float sinn = sin(re*dz);

          float dre = cuCrealf(data->mat[flat_ind]);
          float dim = cuCimagf(data->mat[flat_ind]);

          re = att * (dre * coss - dim * sinn);
          im = att * (dre * sinn + dim * coss);

          model->mat[flat_ind] = cuCaddf(model->mat[flat_ind], make_cuFloatComplex(re, im));
        }
      }
    }
  }
}