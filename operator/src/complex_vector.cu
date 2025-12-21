#include <complex_vector.h>
#include <cuComplex.h>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/functional.h>      // for thrust::plus, thrust::multiplies
#include <thrust/execution_policy.h>
#include <thrust/complex.h>         // <--- CRITICAL: Defines operators for complex
#include <thrust/iterator/constant_iterator.h> // <--- For scaling

// Helper alias to make casting cleaner
using Complex = thrust::complex<float>;

void complex_vector::add(complex_vector* vec) {
    // 1. Cast pointers to Thrust's C++ complex type
    // This is safe because they have the exact same memory layout (2 floats)
    thrust::device_ptr<Complex> ptr1(reinterpret_cast<Complex*>(this->mat));
    thrust::device_ptr<Complex> ptr2(reinterpret_cast<Complex*>(vec->mat));

    // 2. Use standard thrust::plus
    thrust::transform(
        thrust::cuda::par.on(stream),
        ptr1,               // Input 1
        ptr1 + nelem,       // End
        ptr2,               // Input 2
        ptr1,               // Output (In-place)
        thrust::plus<Complex>() // <--- Standard Operator!
    );
}

void complex_vector::scale(float s) {
    // 1. Cast pointer
    thrust::device_ptr<Complex> ptr(reinterpret_cast<Complex*>(this->mat));

    // 2. Create a "virtual" iterator that always returns 's'
    // We treat 's' as a complex number (s + 0j)
    Complex scalar_val(s, 0.0f);
    thrust::constant_iterator<Complex> scale_it(scalar_val);

    // 3. Use standard thrust::multiplies
    thrust::transform(
        thrust::cuda::par.on(stream),
        ptr,                // Input 1 (Vector)
        ptr + nelem,        // End
        scale_it,        // Input 2 (Virtual Vector of scalars)
        ptr,                // Output
        thrust::multiplies<Complex>() // <--- Standard Operator!
    );
}

std::pair<float, float> complex_vector::getMinMax() {
    // 1. Cast complex pointer to raw float pointer
    float* raw_ptr = reinterpret_cast<float*>(mat);

    // 2. Wrap in Thrust device pointer
    thrust::device_ptr<float> dev_ptr(raw_ptr);

    // 3. Find min/max over 2 * nelem (Real + Imag parts)
    // 'par.on(stream)' ensures it queues into your CUDA stream
    auto result = thrust::minmax_element(
        thrust::cuda::par.on(stream), 
        dev_ptr, 
        dev_ptr + (2 * nelem)
    );

    // 4. Dereference iterators to get values
    // Note: This dereference causes a blocking synchronization to read the value to CPU.
    // For strictly async pipelines, see the note below.
    float min_val = *result.first;
    float max_val = *result.second;

    return {min_val, max_val};
  }

