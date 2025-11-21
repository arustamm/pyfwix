#include <RefSampler.h>
#include <numeric>
#include <algorithm>
#include <functional>
#include <vector>
#include <utility>
#include "opencv2/core.hpp"
#include <tbb/tbb.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

using namespace SEP;
using namespace std::placeholders;

RefSampler::RefSampler(std::shared_ptr<hypercube> slow_hyper, std::shared_ptr<paramObj> par) {
	padx = static_cast<size_t>(par->getInt("padx", 0));
	pady = static_cast<size_t>(par->getInt("pady", 0));

	_nref_ = static_cast<size_t>(par->getInt("nref"));
	_nx_ = static_cast<size_t>(slow_hyper->getAxis(1).n);
	_ny_ = static_cast<size_t>(slow_hyper->getAxis(2).n);
	_nw_ = static_cast<size_t>(slow_hyper->getAxis(3).n);
	_nz_ = static_cast<size_t>(slow_hyper->getAxis(4).n);

	ref_labels.resize(boost::extents[_nz_][_nw_][_ny_ + pady][_nx_ + padx]);
	slow_ref.resize(boost::extents[_nz_][_nref_][_nw_]);

	is_sampled.resize(_nz_);

	CHECK_CUDA_ERROR(cudaHostRegister(slow_ref.data(), sizeof(std::complex<float>)*slow_ref.num_elements(), cudaHostRegisterDefault));
	CHECK_CUDA_ERROR(cudaHostRegister(ref_labels.data(), sizeof(int)*ref_labels.num_elements(), cudaHostRegisterDefault));
};

RefSampler::RefSampler(const std::shared_ptr<complex4DReg>& slow, std::shared_ptr<paramObj> par) : RefSampler(slow->getHyper(), par) {
	kmeans_sample(slow);
};

void RefSampler::kmeans_sample(const std::shared_ptr<complex4DReg>& slow) {
	tbb::parallel_for(tbb::blocked_range<size_t>(0,_nz_),
		[=](const tbb::blocked_range<size_t> &r) {
		for (size_t iz=r.begin(); iz < r.end(); iz++) {
			sample_at_depth(slow, iz);
		}
	});
}

void RefSampler::sample_at_depth(std::shared_ptr<complex4DReg> slow, size_t iz) {
	tbb::parallel_for(tbb::blocked_range<size_t>(0,_nw_),
	[=](const tbb::blocked_range<size_t> &r) {
		for (size_t iw=r.begin(); iw < r.end(); iw++) {
			size_t offset = (iw + iz*_nw_)*_nx_*_ny_;
			std::complex<float>* ptr_slow_ref = slow->getVals() + offset;
			// prepare opencv matrices for processing
			cv::Mat_<std::complex<float>> slow_slice(_nx_*_ny_, 1, ptr_slow_ref);
			cv::Mat_<int> labels(_nx_*_ny_, 1);
			cv::Mat_<std::complex<float>> centers(_nref_, 1);
			// stopping criteria
			cv::TermCriteria criteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 100, 1e-3);
			// compute centers & labels
			double obj = cv::kmeans(slow_slice, _nref_, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);
			// copy to slow_ref array
			for (int iref=0; iref < _nref_; ++iref) {
				std::complex<float> sref = centers.at<std::complex<float>>(iref, 0);
				slow_ref[iz][iref][iw] = sref;
			}
			// copy labels to ref_labels array
			for (int iy=0; iy < _ny_; ++iy) {
				for (int ix=0; ix < _nx_; ++ix) {
						size_t flat_index = ix + iy*_nx_;
						ref_labels[iz][iw][iy][ix] = labels.at<int>(flat_index);
				}
		}
			// copy labels to padded region
			for (int iy=_ny_; iy < _ny_ + pady; ++iy) {
				for (int ix=0; ix < _nx_; ++ix) {
					ref_labels[iz][iw][iy][ix] = ref_labels[iz][iw][_ny_-1][ix];
				}
			}
			// copy labels to padded region
			for (int iy=0; iy < _ny_ + pady; ++iy) {
				for (int ix=_nx_; ix < _nx_ + padx; ++ix) {
					ref_labels[iz][iw][iy][ix] = ref_labels[iz][iw][iy][_nx_-1];
				}
			}
		}
	});
	is_sampled[iz] = true;
}

std::future<void> RefSampler::sample_at_depth_async(std::shared_ptr<complex4DReg> slow, size_t iz) {
	return std::async(std::launch::async, [this, slow, iz]() {
			sample_at_depth(slow, iz);
	});
}

