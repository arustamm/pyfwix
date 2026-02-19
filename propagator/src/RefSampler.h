#pragma once
#include <float1DReg.h>
#include <float2DReg.h>
#include "complex2DReg.h"
#include "complex3DReg.h"
#include "complex1DReg.h"
#include "paramObj.h"
#include "boost/multi_array.hpp"
#include  "opencv2/core.hpp"
#include <future>
#include "complex_vector.h"

namespace SEP {


class RefSampler

	{
	public:

		RefSampler(std::shared_ptr<hypercube> slow_hyper, std::shared_ptr<paramObj> par);
		RefSampler(const std::shared_ptr<complex4DReg>& slow, std::shared_ptr<paramObj> par);

		~RefSampler() {
			// CHECK_CUDA_ERROR(cudaHostUnregister(slow_ref.data()));
			// CHECK_CUDA_ERROR(cudaHostUnregister(ref_labels.data()));
		}

		inline std::complex<float>* get_ref_slow(size_t iz, size_t iref) {
			if (!is_sampled[iz]) std::runtime_error("RefSampler: slow not sampled at depth iz");
			return slow_ref.data() + (iref + iz*_nref_)*_nw_;
		}
		// Return pointers to the specific depth slice in the 4D array
		int* get_labels_low(size_t iz) { 
            // boost::multi_array slices don't have .data(), so we address the first element
            return &ref_labels_low[iz][0][0][0]; 
        }

        int* get_labels_high(size_t iz) { 
            return &ref_labels_high[iz][0][0][0]; 
        }

        float* get_weights(size_t iz) { 
            return &ref_weights[iz][0][0][0]; 
        }

		void sample_at_depth(std::shared_ptr<complex4DReg> slow, size_t iz);
		std::future<void> sample_at_depth_async(std::shared_ptr<complex4DReg> slow, size_t iz);

		size_t _nx_, _ny_, _nref_, _nz_, _nw_, padx, pady;
		
	private:
		
		void kmeans_sample(const std::shared_ptr<complex4DReg>& slow);
		
		boost::multi_array<int, 4> ref_labels_low, ref_labels_high;
		boost::multi_array<float, 4> ref_weights;

		boost::multi_array<std::complex<float>, 3> slow_ref;
		std::vector<bool> is_sampled;

		

	};
}
