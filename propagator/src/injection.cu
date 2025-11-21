#include <complex_vector.h>
#include <prop_kernels.cuh>
#include <cuComplex.h>
#include <KernelLauncher.cuh>
#include <KernelLauncher.cu>

__global__ void inj_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
  const float* __restrict__ cx, const float* __restrict__ cy, const float* __restrict__ cz, const int* __restrict__ ids, float oz, float dz, int iz_to_inject) {

  int NX = data->n[0];
  int NY = data->n[1];
  int NW = data->n[2];
  int NS = data->n[3];
  
  int mNW = model->n[0];
  int NTRACE = model->n[1];
  int mdims[] = {NTRACE,mNW};

  float OX = data->o[0];
  float OY = data->o[1];
  float DX = data->d[0];
  float DY = data->d[1];
  int dims[] = {NS, NW, NY, NX};

  int iw0 = threadIdx.x + blockDim.x*blockIdx.x;
  int itrace0 = threadIdx.y + blockDim.y*blockIdx.y;

  int jw = blockDim.x * gridDim.x;
  int jtrace = blockDim.y * gridDim.y;

  cuFloatComplex w[4]; 

  for (int itrace=itrace0; itrace < NTRACE; itrace += jtrace) {
    int iy = floorf((cy[itrace]-OY)/DY);
    float y = OY + iy*DY;
    float ly = 1.f - (cy[itrace] - y) / DY;

    int ix = floorf((cx[itrace]-OX)/DX);
    float x = OX + ix*DX;
    float lx = 1.f - (cx[itrace] - x) / DX;

    int iz = floorf((cz[itrace]-oz)/dz);
    float z = oz + iz*dz;
    float lz = 1.f - (cz[itrace] - z) / dz;

    if (ix < 0 || ix >= NX - 1 || iy < 0 || iy >= NY - 1) {
      continue;
    }

    // Skip if current Z level is not relevant for this trace's interpolation
    if (iz_to_inject != iz && iz_to_inject != iz + 1) {
      continue;
    }

    int id = ids[itrace];

    // Compute weights based on whether we're at the lower or upper Z level
    if (iz_to_inject == iz) {
      // We're at the lower Z level
      w[0] = make_cuFloatComplex(lz * lx * ly, 0.0f);
      w[1] = make_cuFloatComplex(lz * (1 - lx) * ly, 0.0f);
      w[2] = make_cuFloatComplex(lz * lx * (1 - ly), 0.0f);
      w[3] = make_cuFloatComplex(lz * (1 - lx) * (1 - ly), 0.0f);
    } 
    else {
      // We're at the upper Z level (iz_current == iz + 1)
      w[0] = make_cuFloatComplex((1 - lz) * lx * ly, 0.0f);
      w[1] = make_cuFloatComplex((1 - lz) * (1 - lx) * ly, 0.0f);
      w[2] = make_cuFloatComplex((1 - lz) * lx * (1 - ly), 0.0f);
      w[3] = make_cuFloatComplex((1 - lz) * (1 - lx) * (1 - ly), 0.0f);
    }
    
    for (int iw=iw0; iw < mNW; iw += jw) {
      int idx_ind[] = {itrace, iw};
      int idx_cy0cx0[] = {id, iw, iy, ix};
      int idx_cy0cx1[] = {id, iw, iy, ix + 1};
      int idx_cy1cx0[] = {id, iw, iy + 1, ix};
      int idx_cy1cx1[] = {id, iw, iy + 1, ix + 1};

      // Convert indices to flat indices
      size_t ind = ND_TO_FLAT(idx_ind, mdims);
      size_t cy0cx0 = ND_TO_FLAT(idx_cy0cx0, dims);
      size_t cy0cx1 = ND_TO_FLAT(idx_cy0cx1, dims);
      size_t cy1cx0 = ND_TO_FLAT(idx_cy1cx0, dims);
      size_t cy1cx1 = ND_TO_FLAT(idx_cy1cx1, dims);

      cuFloatComplex val = model->mat[ind];

      data->mat[cy0cx0] = cuCaddf(data->mat[cy0cx0], cuCmulf(w[0], val)); 
      data->mat[cy0cx1] = cuCaddf(data->mat[cy0cx1], cuCmulf(w[1], val)); 
      data->mat[cy1cx0] = cuCaddf(data->mat[cy1cx0], cuCmulf(w[2], val)); 
      data->mat[cy1cx1] = cuCaddf(data->mat[cy1cx1], cuCmulf(w[3], val));

    }
  }
};

__global__ void inj_adjoint(complex_vector* __restrict__ model, complex_vector* __restrict__ data, 
  const float* __restrict__ cx, const float* __restrict__ cy, const float* __restrict__ cz, const int* __restrict__ ids, float oz, float dz, int iz_to_inject) {

  int NX = data->n[0];
  int NY = data->n[1];
  int NW = data->n[2];
  int NS = data->n[3];

  int mNW = model->n[0];
  int NTRACE = model->n[1];
  int mdims[] = {NTRACE,mNW};

  float OX = data->o[0];
  float OY = data->o[1];
  float DX = data->d[0];
  float DY = data->d[1];
  int dims[] = {NS, NW, NY, NX};

  int iw0 = threadIdx.x + blockDim.x*blockIdx.x;
  int itrace0 = threadIdx.y + blockDim.y*blockIdx.y;

  int jw = blockDim.x * gridDim.x;
  int jtrace = blockDim.y * gridDim.y;
  
  cuFloatComplex w[4];

  for (int itrace=itrace0; itrace < NTRACE; itrace += jtrace) {
    int iy = floorf((cy[itrace]-OY)/DY);
    float y = OY + iy*DY;
    float ly = 1.f - (cy[itrace] - y) / DY;

    int ix = floorf((cx[itrace]-OX)/DX);
    float x = OX + ix*DX;
    float lx = 1.f - (cx[itrace] - x) / DX;

    int iz = floorf((cz[itrace]-oz)/dz);
    float z = oz + iz*dz;
    float lz = 1.f - (cz[itrace] - z) / dz;

    if (ix < 0 || ix >= NX - 1 || iy < 0 || iy >= NY - 1) {
      continue;
    }

    // Skip if current Z level is not relevant for this trace's interpolation
    if (iz_to_inject != iz && iz_to_inject != iz + 1) {
      continue;
    }

    int id = ids[itrace];

    // Compute weights based on whether we're at the lower or upper Z level
    if (iz_to_inject == iz) {
      // We're at the lower Z level
      w[0] = make_cuFloatComplex(lz * lx * ly, 0.0f);
      w[1] = make_cuFloatComplex(lz * (1 - lx) * ly, 0.0f);
      w[2] = make_cuFloatComplex(lz * lx * (1 - ly), 0.0f);
      w[3] = make_cuFloatComplex(lz * (1 - lx) * (1 - ly), 0.0f);
    } 
    else {
      // We're at the upper Z level (iz_current == iz + 1)
      w[0] = make_cuFloatComplex((1 - lz) * lx * ly, 0.0f);
      w[1] = make_cuFloatComplex((1 - lz) * (1 - lx) * ly, 0.0f);
      w[2] = make_cuFloatComplex((1 - lz) * lx * (1 - ly), 0.0f);
      w[3] = make_cuFloatComplex((1 - lz) * (1 - lx) * (1 - ly), 0.0f);
    }
    
    for (int iw=iw0; iw < mNW; iw += jw) {

      int idx_ind[] = {itrace, iw};
      int idx_cy0cx0[] = {id, iw, iy, ix};
      int idx_cy0cx1[] = {id, iw, iy, ix + 1};
      int idx_cy1cx0[] = {id, iw, iy + 1, ix};
      int idx_cy1cx1[] = {id, iw, iy + 1, ix + 1};
      
      // Convert indices to flat indices
      size_t ind = ND_TO_FLAT(idx_ind, mdims);
      size_t cy0cx0 = ND_TO_FLAT(idx_cy0cx0, dims);
      size_t cy0cx1 = ND_TO_FLAT(idx_cy0cx1, dims);
      size_t cy1cx0 = ND_TO_FLAT(idx_cy1cx0, dims);
      size_t cy1cx1 = ND_TO_FLAT(idx_cy1cx1, dims);

      cuFloatComplex val = cuCmulf(data->mat[cy0cx0], w[0]);
      val = cuCaddf(val, cuCmulf(data->mat[cy0cx1], w[1])); 
      val = cuCaddf(val, cuCmulf(data->mat[cy1cx0], w[2])); 
      val = cuCaddf(val, cuCmulf(data->mat[cy1cx1], w[3])); 

      model->mat[ind] = cuCaddf(model->mat[ind], val);

    }
  }
};