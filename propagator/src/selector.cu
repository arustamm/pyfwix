#include <complex_vector.h>
#include <prop_kernels.cuh>
#include <cuComplex.h>
#include <KernelLauncher.cuh>
#include <KernelLauncher.cu>

// NEW KERNEL: Weighted Selection / Interpolation
__global__ void select_forward(
    complex_vector* __restrict__ model, 
    complex_vector* __restrict__ data, 
    int current_ref_idx, 
    const int* __restrict__ labels_low,
    const int* __restrict__ labels_high,
    const float* __restrict__ weights
) {
  const int NX = model->n[0];
  const int NY = model->n[1];
  const int NW = model->n[2];
  const int NS = model->n[3];
  
  // Calculate linear thread ID
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_threads = gridDim.x * blockDim.x;
  const int total_elements = NX * NY * NW * NS;
  
  // Grid Stride Loop
  for (int idx = tid; idx < total_elements; idx += total_threads) {
    // 1. Map 4D flat index (idx) to 3D spatial index (i)
    // idx covers (is, iw, iy, ix), but labels are only (iw, iy, ix)
    // Formula: i = ix + iy*NX + iw*NX*NY
    int ix = idx % NX;
    int iy = (idx / NX) % NY;
    int iw = (idx / (NX * NY)) % NW;
    
    // Spatial index for looking up weights/labels
    size_t i = ix + (iy + iw * NY) * NX;
          
    // 2. Determine Contribution Weight
    float contribution = 0.0f;
    bool active = false;

    // Check Lower Bracket: Weight = (1.0 - alpha)
    if (labels_low[i] == current_ref_idx) {
        contribution += (1.0f - weights[i]);
        active = true;
    }

    // Check Upper Bracket: Weight = alpha
    if (labels_high[i] == current_ref_idx) {
        contribution += weights[i];
        active = true;
    }

    // 3. Apply Weight if Active
    // If active is true, this pixel belongs (fully or partially) to this reference velocity
    if (active && contribution > 1e-6f) {
        // Linear Interpolation: val = input * weight
        cuComplex weighted_val = cuCmulf(model->mat[idx], make_cuComplex(contribution, 0.0f));
        
        // Accumulate: output += val
        data->mat[idx] = cuCaddf(data->mat[idx], weighted_val); 
    }
  }
};