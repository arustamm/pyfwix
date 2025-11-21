#include <complex_vector.h>
#include <complex_vector.cuh>
#include <iostream>

complex_vector* make_complex_vector(const std::shared_ptr<hypercube>& hyper, dim3 grid, dim3 block, cudaStream_t stream) {
  complex_vector* vec;
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&vec), sizeof(complex_vector)));

  vec->set_grid_block(grid, block);
  vec->stream = stream;

  int nelem = vec->nelem = hyper->getN123();
  int ndim = vec->ndim = hyper->getNdim();
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&vec->n), sizeof(int) * ndim));
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&vec->d), sizeof(float) * ndim));
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&vec->o), sizeof(float) * ndim));
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&vec->mat), sizeof(cuFloatComplex) * nelem));

  for (int i=0; i < ndim; ++i) {
    vec->n[i] = hyper->getAxis(i+1).n;
    vec->d[i] = hyper->getAxis(i+1).d;
    vec->o[i] = hyper->getAxis(i+1).o;
  }
  vec->allocated = true;

  return vec;
};

complex_vector* complex_vector::cloneSpace() {
  complex_vector* vec;
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&vec), sizeof(complex_vector)));

  vec->set_grid_block(this->_grid_, this->_block_);
  vec->stream = stream;

  int nelem = vec->nelem = this->nelem;
  int ndim = vec->ndim = this->ndim;
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&vec->n), sizeof(int) * ndim));
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&vec->d), sizeof(float) * ndim));
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&vec->o), sizeof(float) * ndim));
  CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&vec->mat), sizeof(cuFloatComplex) * nelem));

  for (int i=0; i < ndim; ++i) {
    vec->n[i] = this->n[i];
    vec->d[i] = this->d[i];
    vec->o[i] = this->o[i];
  }
  vec->allocated = true;

  return vec;
};

void complex_vector::add(complex_vector* vec){
  launch_add(this, vec, _grid_, _block_, this->stream);
}

complex_vector*  complex_vector::make_view(int start, int end) {
  complex_vector* view;
  CHECK_CUDA_ERROR(cudaMallocManaged(&view, sizeof(complex_vector)));

  view->set_grid_block(_grid_, _block_);
  view->set_stream(stream);

  // Calculate the size of the new vector
  view->ndim = this->ndim; // Keep the same number of dimensions
  
  // Calculate nelem based on the range
  view->nelem = 1; 
  for (int i = 0; i < view->ndim - 1; ++i) {
    view->nelem *= this->n[i];
  }
  view->nelem *= end - start; // Account for the range of slices

  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&view->n), sizeof(int) * view->ndim));
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&view->d), sizeof(float) * view->ndim));
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&view->o), sizeof(float) * view->ndim));

  // Copy dimensions, adjusting the slowest dimension
  for (int i = 0; i < view->ndim - 1; ++i) {
    view->n[i] = this->n[i];
    view->d[i] = this->d[i];
    view->o[i] = this->o[i];
  }
  view->n[view->ndim - 1] = end - start; // Set the size of the slowest dimension
  view->d[view->ndim - 1] = this->d[view->ndim - 1]; 
  view->o[view->ndim - 1] = this->o[view->ndim - 1] + start * this->d[view->ndim - 1]; // Adjust the origin

  view->allocated = false;

  // Calculate the offset for the starting slice
  int offset = start;
  for (int i = 0; i < this->ndim - 1; ++i) {
    offset *= this->n[i];
  }
  view->mat = this->mat + offset;

  return view;
}

complex_vector*  complex_vector::make_slice() {
  complex_vector* slice;
  CHECK_CUDA_ERROR(cudaMallocManaged(&slice, sizeof(complex_vector)));

  slice->set_grid_block(_grid_, _block_);
  slice->set_stream(stream);

  // Calculate the size of the new vector
  slice->ndim = this->ndim - 1; // Remove last dimension
  
  // Calculate nelem based on the range
  slice->nelem = 1; 
  for (int i = 0; i < slice->ndim; ++i) {
    slice->nelem *= this->n[i];
  }

  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&slice->n), sizeof(int) * slice->ndim));
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&slice->d), sizeof(float) * slice->ndim));
  CHECK_CUDA_ERROR(cudaMallocManaged(reinterpret_cast<void **>(&slice->o), sizeof(float) * slice->ndim));

  // Copy dimensions, adjusting the slowest dimension
  for (int i = 0; i < slice->ndim; ++i) {
    slice->n[i] = this->n[i];
    slice->d[i] = this->d[i];
    slice->o[i] = this->o[i];
  }

  slice->allocated = false;
  slice->mat = this->mat;

  return slice;
}

// Const version (cannot modify the object)
// const complex_vector* complex_vector::make_const_view() {
//   return const_cast<const complex_vector*>(this->make_view()); 
// }

void complex_vector::slice_at(complex_vector* slice, int index) {
  if (slice->allocated) throw std::runtime_error("The provided complex_vector is not a slice.");
  if (slice->ndim != this->ndim - 1) throw std::runtime_error("The provided complex_vector is not a valid slice of the current vector.");
  // Calculate the offset in the original data
  size_t offset = static_cast<size_t>(index) * slice->nelem;
  slice->mat = this->mat + offset; 
}

// void complex_vector::view_at(const complex_vector* view, int index) {
//   if (view->allocated) throw std::runtime_error("The provided complex_vector is not a view!");
//   // Calculate the offset in the original data
//   int offset = index * view->nelem;
//   const_cast<complex_vector*>(view)->mat = this->mat + offset; 
// }





