#include <complex_vector.h>
#include <prop_kernels.cuh>
#include <cuComplex.h>
#include <KernelLauncher.cuh>
#include <KernelLauncher.cu>

__global__ void select_forward(complex_vector* __restrict__ model, complex_vector* __restrict__ data, int value, const int* __restrict__ labels) {

  const int NX = model->n[0];
  const int NY = model->n[1];
  const int NW = model->n[2];
  const int NS = model->n[3];
  const int dims[] = {NS, NW, NY, NX};
  
  // Calculate linear thread ID in the grid
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_threads = gridDim.x * blockDim.x;
  
  // Each thread processes multiple elements across the entire 4D domain
  const int total_elements = NX * NY * NW * NS;
  
  // Process elements with stride equal to total number of threads
  for (int idx = tid; idx < total_elements; idx += total_threads) {
// Convert linear index to 4D coordinates
    int ix = idx % NX;
    int iy = (idx / NX) % NY;
    int iw = (idx / (NX * NY)) % NW;
    int is = idx / (NX * NY * NW);
    size_t i = ix + (iy + iw*NY)*NX;
          
    if (labels[i] == value) {
      // for (int is=0; is < NS; ++is) {
        // int nd_ind[] = {is, iw, iy, ix};
        // size_t ind = ND_TO_FLAT(nd_ind, dims);
        data->mat[idx] = cuCaddf(data->mat[idx], model->mat[idx]); 
    // }
    }
  }
};
