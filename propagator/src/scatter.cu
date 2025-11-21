#include <complex_vector.h>
#include <prop_kernels.cuh>
#include <cuComplex.h>
#include <KernelLauncher.cuh>

__device__ cuFloatComplex cuCpowf(cuFloatComplex base, float real_exp) {
    float base_real = cuCrealf(base);
    float base_imag = cuCimagf(base);
    
    // For z^r where r is real: z^r = |z|^r * exp(i*r*arg(z))
    float magnitude = sqrtf(base_real * base_real + base_imag * base_imag);
    float argument = atan2f(base_imag, base_real);
    
    float result_magnitude = powf(magnitude, real_exp);
    float result_argument = real_exp * argument;
    
    float result_real = result_magnitude * cosf(result_argument);
    float result_imag = result_magnitude * sinf(result_argument);
    
    return make_cuFloatComplex(result_real, result_imag);
}

__global__ void mult_kxky(complex_vector* __restrict__ model, complex_vector* __restrict__ data,
  const float* __restrict__ w, const float* __restrict__ kx, const float* __restrict__ ky, int it) {

  int NX = model->n[0];
  int NY = model->n[1];
  int NW = model->n[2];
  int NS = model->n[3];
  int dims[] = {NS, NW, NY, NX};

  int ix0 = threadIdx.x + blockDim.x * blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y * blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z * blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int iw = iw0; iw < NW; iw += jw) {
    for (int iy = iy0; iy < NY; iy += jy) {
      for (int ix = ix0; ix < NX; ix += jx) {
        
        float k_mag_sq = kx[ix] * kx[ix] + ky[iy] * ky[iy];
        float pow_factor = powf(sqrtf(k_mag_sq) / w[iw], 2 * it);

        for (int is = 0; is < NS; is++) {
          int nd_ind[] = {is, iw, iy, ix};
          size_t flat_ind = ND_TO_FLAT(nd_ind, dims);

          data->mat[flat_ind] = cuCmulf(model->mat[flat_ind], make_cuFloatComplex(pow_factor, 0.0f));
        }
      }
    }
  }
}

__global__ void slow_scale_fwd(complex_vector* __restrict__ model, complex_vector* __restrict__ data,
  const complex_vector* __restrict__ slow_slice, float coef, 
  int it, float eps) {

  int NX = model->n[0];
  int NY = model->n[1];
  int NW = model->n[2];
  int NS = model->n[3];
  int dims[] = {NS, NW, NY, NX};

  int ix0 = threadIdx.x + blockDim.x * blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y * blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z * blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int iw = iw0; iw < NW; iw += jw) {
    for (int iy = iy0; iy < NY; iy += jy) {
      for (int ix = ix0; ix < NX; ix += jx) {
        
        size_t slow_ind = ix + NX*(iy + iw*NY);
        cuFloatComplex slow_val = slow_slice->mat[slow_ind];
        
        // Apply epsilon damping
        float slow_real = cuCrealf(slow_val);
        float slow_imag = cuCimagf(slow_val) - eps * slow_real;
        cuFloatComplex slow_modified = make_cuFloatComplex(slow_real, slow_imag);
        
        // Compute c = coef[it] * slow^(-0.5-it)
        cuFloatComplex c = cuCmulf(make_cuFloatComplex(coef,0.f), cuCpowf(slow_modified, -0.5f - it));

        for (int is = 0; is < NS; is++) {
          int nd_ind[] = {is, iw, iy, ix};
          size_t flat_ind = ND_TO_FLAT(nd_ind, dims);

          data->mat[flat_ind] = cuCmulf(c, model->mat[flat_ind]);
        }
      }
    }
  }
}

__global__ void slow_scale_adj(complex_vector* __restrict__ model, complex_vector* __restrict__ data,
  const complex_vector* __restrict__ slow_slice, float coef, 
  int it, float eps) {

  int NX = model->n[0];
  int NY = model->n[1];
  int NW = model->n[2];
  int NS = model->n[3];
  int dims[] = {NS, NW, NY, NX};

  int ix0 = threadIdx.x + blockDim.x * blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y * blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z * blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int iw = iw0; iw < NW; iw += jw) {
    for (int iy = iy0; iy < NY; iy += jy) {
      for (int ix = ix0; ix < NX; ix += jx) {
        
        size_t slow_ind = ix + NX*(iy + iw*NY);
        cuFloatComplex slow_val = slow_slice->mat[slow_ind];
        
        // Apply epsilon damping
        float slow_real = cuCrealf(slow_val);
        float slow_imag = cuCimagf(slow_val) - eps * slow_real;
        cuFloatComplex slow_modified = make_cuFloatComplex(slow_real, slow_imag);
        
        // Compute c = coef[it] * slow^(-0.5-it)
        cuFloatComplex c = cuCmulf(make_cuFloatComplex(coef,0.f), cuCpowf(slow_modified, -0.5f - it));

        for (int is = 0; is < NS; is++) {
          int nd_ind[] = {is, iw, iy, ix};
          size_t flat_ind = ND_TO_FLAT(nd_ind, dims);

          model->mat[flat_ind] = cuCmulf(cuConjf(c), data->mat[flat_ind]);
        }
      }
    }
  }
}

__global__ void scale_by_iw_fwd(complex_vector* __restrict__ model, complex_vector* __restrict__ data,
  const float* __restrict__ w, float dz) {

  int NX = model->n[0];
  int NY = model->n[1];
  int NW = model->n[2];
  int NS = model->n[3];
  int dims[] = {NS, NW, NY, NX};

  int ix0 = threadIdx.x + blockDim.x * blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y * blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z * blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int iw = iw0; iw < NW; iw += jw) {
    float factor = - 0.5f * w[iw] * dz;
    
    for (int iy = iy0; iy < NY; iy += jy) {
      for (int ix = ix0; ix < NX; ix += jx) {
        
        for (int is = 0; is < NS; is++) {
          int nd_ind[] = {is, iw, iy, ix};
          size_t flat_ind = ND_TO_FLAT(nd_ind, dims);

          cuFloatComplex model_val = model->mat[flat_ind];
          float re = -factor * cuCimagf(model_val);
          float im = factor * cuCrealf(model_val);
          
          cuFloatComplex increment = make_cuFloatComplex(re, im);
          data->mat[flat_ind] = cuCaddf(data->mat[flat_ind], increment);
        }
      }
    }
  }
}

__global__ void scale_by_iw_adj(complex_vector* __restrict__ model, complex_vector* __restrict__ data,
  const float* __restrict__ w, float dz) {

  int NX = model->n[0];
  int NY = model->n[1];
  int NW = model->n[2];
  int NS = model->n[3];
  int dims[] = {NS, NW, NY, NX};

  int ix0 = threadIdx.x + blockDim.x * blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y * blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z * blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int iw = iw0; iw < NW; iw += jw) {
    float factor = 0.5f * w[iw] * dz;
    
    for (int iy = iy0; iy < NY; iy += jy) {
      for (int ix = ix0; ix < NX; ix += jx) {
        
        for (int is = 0; is < NS; is++) {
          int nd_ind[] = {is, iw, iy, ix};
          size_t flat_ind = ND_TO_FLAT(nd_ind, dims);

          cuFloatComplex data_val = data->mat[flat_ind];
          float re = -factor * cuCimagf(data_val);
          float im = factor * cuCrealf(data_val);
          
          cuFloatComplex increment = make_cuFloatComplex(re, im);
          model->mat[flat_ind] = cuCaddf(model->mat[flat_ind], increment);
        }
      }
    }
  }
}

__global__ void pad_fwd(
  complex_vector* __restrict__ model, //unpadded
  complex_vector* __restrict__ data //padded
) {

  int NX = data->n[0];
  int NY = data->n[1];
  int NW = data->n[2];
  int nx = model->n[0];
  int ny = model->n[1];
  int dims[] = {NW, NY, NX};
  int in_dims[] = {NW, ny, nx};
  
  int ix0 = threadIdx.x + blockDim.x * blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y * blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z * blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int iw = iw0; iw < NW; iw += jw) {    
    for (int iy = iy0; iy < NY; iy += jy) {
      for (int ix = ix0; ix < NX; ix += jx) {
        
        int out_nd_ind[] = {iw, iy, ix};
        size_t out_ind = ND_TO_FLAT(out_nd_ind, dims);
        // Determine input indices with padding by extending last values
        int ix_in = min(ix, nx - 1);
        int iy_in = min(iy, ny - 1);
        int in_nd_ind[] = {iw, iy_in, ix_in};
        size_t in_ind = ND_TO_FLAT(in_nd_ind, in_dims);

        data->mat[out_ind] = model->mat[in_ind];
      }
    }
  }
}

__global__ void pad_adj(
  complex_vector* __restrict__ model, //unpadded
  complex_vector* __restrict__ data //padded
) {

  int NX = data->n[0];
  int NY = data->n[1];
  int NW = data->n[2];
  int nx = model->n[0];
  int ny = model->n[1];
  int dims[] = {NW, NY, NX};
  int in_dims[] = {NW, ny, nx};
  
  int ix0 = threadIdx.x + blockDim.x * blockIdx.x;
  int iy0 = threadIdx.y + blockDim.y * blockIdx.y;
  int iw0 = threadIdx.z + blockDim.z * blockIdx.z;

  int jx = blockDim.x * gridDim.x;
  int jy = blockDim.y * gridDim.y;
  int jw = blockDim.z * gridDim.z;

  for (int iw = iw0; iw < NW; iw += jw) {    
    for (int iy = iy0; iy < NY; iy += jy) {
      for (int ix = ix0; ix < NX; ix += jx) {
        
        int out_nd_ind[] = {iw, iy, ix};
        size_t out_ind = ND_TO_FLAT(out_nd_ind, dims);
        // Determine input indices with padding by extending last values
        int ix_in = min(ix, nx - 1);
        int iy_in = min(iy, ny - 1);
        int in_nd_ind[] = {iw, iy_in, ix_in};
        size_t in_ind = ND_TO_FLAT(in_nd_ind, in_dims);

        model->mat[in_ind] = cuCaddf(model->mat[in_ind], data->mat[out_ind]);
      }
    }
  }
}

    

