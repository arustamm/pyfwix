#pragma once
#include <hypercube.h>
#include <complex>
#include <cuda_runtime.h>
#include <iostream>
#include <cuComplex.h>
#include <unordered_map>

// thanks to Google's AI Bard
#define ND_TO_FLAT(idx, dims) ( \
    ({ \
        size_t _flat_idx = 0; \
        size_t _dim_product = 1; \
        for (int i = sizeof(dims) / sizeof(dims[0]) - 1; i >= 0; --i) { \
            _flat_idx += (idx[i]) * _dim_product; \
            _dim_product *= dims[i]; \
        } \
        _flat_idx; \
    }) \
)

#define CHECK_CUDA_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
};

using namespace SEP;

typedef struct complex_vector
{
    cuFloatComplex* mat;
    bool allocated = false;
    int* n;
    float* d;
    float* o;
    int nelem, ndim;
    // for kernels
    dim3 _grid_, _block_;
    cudaStream_t stream = 0;

    void set_grid_block(dim3 grid, dim3 block) {
      _grid_ = grid.x * grid.y * grid.z;
      _block_ = block.x * block.y * block.z;
    }

    void set_stream(cudaStream_t stream) {
      this->stream = stream;
    }

    void zero() {
      CHECK_CUDA_ERROR(cudaMemsetAsync(mat, 0, sizeof(cuFloatComplex)*nelem, this->stream));
    }

    complex_vector* cloneSpace();

    void add(complex_vector* vec);

    complex_vector* make_view(int start, int end);
    complex_vector* make_slice();
    // const complex_vector* make_const_view();

    // to slice the multi-d array along the last axis and return (ndim-1)-d array
    void slice_at(complex_vector* slice, int index);
    // void view_at(const complex_vector* view, int index);

    ~complex_vector() {
      if (allocated) {
        CHECK_CUDA_ERROR(cudaFree(mat));
        allocated = false;
      }
      CHECK_CUDA_ERROR(cudaFree(n)); 
      CHECK_CUDA_ERROR(cudaFree(d));
      CHECK_CUDA_ERROR(cudaFree(o));
    };
} complex_vector;

complex_vector* make_complex_vector(const std::shared_ptr<hypercube>& hyper, dim3 grid=1, dim3 block=1, cudaStream_t stream = 0);
// create the view (only slice through last axis)
complex_vector* make_view(complex_vector* parent);





