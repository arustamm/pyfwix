#ifndef KMEANS_CUH
#define KMEANS_CUH

#include <cuComplex.h>
#include "complex_vector.h"

// Kernel to compute distances between points and centers
__global__ void computeDistancesKernel(
    const cuFloatComplex* data,
    const cuFloatComplex* centers,
    float* distances,
    int n_points,
    int n_centers);

// Kernel to assign points to their closest center
__global__ void assignPointsKernel(
    const float* distances,
    int* labels,
    int n_points,
    int n_centers);

// Kernel to update center positions
__global__ void updateCentersKernel(
    const cuFloatComplex* data,
    const int* labels,
    cuFloatComplex* centers,
    int* counts,
    int n_points,
    int n_centers);

// Kernel to normalize center positions
__global__ void normalizeCentersKernel(
    cuFloatComplex* centers,
    int* counts,
    int n_centers);

// Kernel to store labels in reference array with padding
__global__ void storeLabelsKernel(
    int* labels,
    int* ref_labels,
    int nx,
    int ny,
    int padx,
    int pady,
    int nx_padded,
    int ny_padded,
    size_t iz,
    size_t iw,
    size_t nw,
    size_t nx_blocks);

// Check if the centers have converged
__global__ void computeDeltasKernel(
    const cuFloatComplex* old_centers, 
    const cuFloatComplex* new_centers, 
    float* d_center_deltas, 
    int n_centers);

// Kernel to store centers in reference slowness array
__global__ void storeCentersKernel(
    cuFloatComplex* centers,
    cuFloatComplex* slow_ref,
    size_t iz,
    size_t iw,
    size_t nref,
    size_t nw);


// Add these declarations:
void launchComputeDistances(cuFloatComplex* d_data, cuFloatComplex* d_centers, 
    float* d_distances, int n_points, int n_centers, int gridSize, int blockSize);

void launchAssignPoints(float* d_distances, int* d_labels, 
    int n_points, int n_centers, int gridSize, int blockSize);

void launchUpdateCenters(cuFloatComplex* d_data, int* d_labels, 
    cuFloatComplex* d_new_centers, int* d_counts, int n_points, int n_centers, int gridSize, int blockSize);

void launchNormalizeCenters(cuFloatComplex* d_new_centers, int* d_counts, 
    int n_centers, int gridSize, int blockSize);

void launchComputeDeltas(cuFloatComplex* d_centers, cuFloatComplex* d_new_centers, 
    float* d_center_deltas, int n_centers, int gridSize, int blockSize);

void launchStoreCenters(cuFloatComplex* d_centers, cuFloatComplex* d_slow_ref,
    size_t iz, size_t iw, size_t nref, size_t nw, dim3 gridSize, dim3 blockSize);

void launchStoreLabels(int* d_labels, int* d_ref_labels,
    int nx, int ny, int padx, int pady, int nx_padded, int ny_padded,
    size_t iz, size_t iw, size_t nw, size_t nx_blocks, dim3 gridSize, dim3 blockSize);

#endif // KMEANS_CUH