#include <complex_vector.h>
#include <prop_kernels.cuh>
#include <cuComplex.h>
#include <KernelLauncher.cuh>
#include <KernelLauncher.cu>

__global__ void taper_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, int tapx, int tapy) {

  int flat_ind;
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
    for (int iy=iy0; iy < NY; iy += jy) {
      for (int ix=ix0; ix < NX; ix += jx) {

        // Check if we need to apply taper
        bool in_taper_region = (ix < tapx) || (ix >= NX-tapx) || 
        (iy < tapy) || (iy >= NY-tapy);

        // Skip calculation if not in taper region
        if (!in_taper_region) {
          // Just copy model to data outside taper region
          int nd_ind[] = {is, iw, iy, ix};
          flat_ind = ND_TO_FLAT(nd_ind, dims);
          data->mat[flat_ind] = cuCaddf(data->mat[flat_ind], model->mat[flat_ind]);
          continue;
        }

        // Calculate taper weight for X dimension
        float weight_x = 1.0f;
        if (ix < tapx && tapx > 0) {
          weight_x = 0.5f * (1.0f - cosf(M_PI * ix / tapx));
        } else if (ix >= NX-tapx && ix < NX) {
          weight_x = 0.5f * (1.0f + cosf(M_PI * (ix - NX + tapx) / tapx));
        }
        
        // Calculate taper weight for Y dimension
        float weight_y = 1.0f;
        if (iy < tapy && tapy > 0) {
          weight_y = 0.5f * (1.0f - cosf(M_PI * iy / tapy));
        } else if (iy >= NY-tapy && iy < NY) {
          weight_y = 0.5f * (1.0f + cosf(M_PI * (iy - NY + tapy) / tapy));
        }
        
        // Combine weights - use minimum for smooth corners
        float weight = weight_x * weight_y;
        
        // Convert 4D index to flat index
        int nd_ind[] = {is, iw, iy, ix};
        flat_ind = ND_TO_FLAT(nd_ind, dims);
        
        // Apply taper to model values and add to data
        float re = cuCrealf(model->mat[flat_ind]) * weight;
        float im = cuCimagf(model->mat[flat_ind]) * weight;
        
        data->mat[flat_ind] = cuCaddf(data->mat[flat_ind], make_cuFloatComplex(re, im));
      }
      }
    }
  }
};
