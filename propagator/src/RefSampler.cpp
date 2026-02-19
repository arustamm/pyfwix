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
    _nref_ = static_cast<size_t>(par->getInt("nref"));
    _nx_ = static_cast<size_t>(slow_hyper->getAxis(1).n);
    _ny_ = static_cast<size_t>(slow_hyper->getAxis(2).n);
    _nw_ = static_cast<size_t>(slow_hyper->getAxis(3).n);
    _nz_ = static_cast<size_t>(slow_hyper->getAxis(4).n);

    padx = par->getInt("padx", 0);
    pady = par->getInt("pady", 0);
    if (padx < 0 || pady < 0) 
        throw std::runtime_error("RefSampler: padx and pady must be >= 0");     

    // Allocations for Interpolation Maps
    // We need 2 index maps (low, high) and 1 float weight map
    ref_labels_low.resize(boost::extents[_nz_][_nw_][_ny_ + pady][_nx_ + padx]);
    ref_labels_high.resize(boost::extents[_nz_][_nw_][_ny_ + pady][_nx_ + padx]);
    ref_weights.resize(boost::extents[_nz_][_nw_][_ny_ + pady][_nx_ + padx]);

    slow_ref.resize(boost::extents[_nz_][_nref_][_nw_]);
    is_sampled.resize(_nz_);
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
    tbb::parallel_for(tbb::blocked_range<size_t>(0, _nw_),
        [=](const tbb::blocked_range<size_t>& r) {
        for (size_t iw = r.begin(); iw < r.end(); iw++) {
            size_t offset = (iw + iz * _nw_) * _nx_ * _ny_;
            std::complex<float>* ptr_slow_ref = slow->getVals() + offset;

            // 1. Prepare Data for K-Means (2 channels: Real, Imag)
            // This ensures Euclidean distance works correctly in complex plane
            cv::Mat points(_nx_ * _ny_, 1, CV_32FC2, ptr_slow_ref);
            
            cv::Mat labels;
            cv::Mat centers;
            
            // Term Criteria
            cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-3);

            // Run K-Means
            double obj = cv::kmeans(points, _nref_, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);

            // =========================================================
            // 2. SORTING (Mandatory for Interpolation)
            // =========================================================
            
            // Create a vector of (Real_Part, Original_Index) to sort
            std::vector<std::pair<float, int>> sorted_indices(_nref_);
            std::vector<std::complex<float>> sorted_centers(_nref_);

            for (int k = 0; k < _nref_; k++) {
                std::complex<float> c = centers.at<std::complex<float>>(k);
                // Sort primarily by Real part (Slowness/Velocity)
                sorted_indices[k] = std::make_pair(c.real(), k);
            }

            // Sort ascending: v_0 < v_1 < ... < v_n
            std::sort(sorted_indices.begin(), sorted_indices.end());

            // Store sorted references
            for (int new_k = 0; new_k < _nref_; new_k++) {
                int old_k = sorted_indices[new_k].second;
                std::complex<float> val = centers.at<std::complex<float>>(old_k);
                slow_ref[iz][new_k][iw] = val;
                sorted_centers[new_k] = val;
            }

            // =========================================================
            // 3. COMPUTE INTERPOLATION BRACKETS & WEIGHTS
            // =========================================================
            
            for (int iy = 0; iy < _ny_; ++iy) {
                for (int ix = 0; ix < _nx_; ++ix) {
                    size_t flat_idx = ix + iy * _nx_;
                    std::complex<float> v_pixel = ptr_slow_ref[flat_idx];
                    float s_real = v_pixel.real();

                    int idx_low = 0;
                    int idx_high = 0;
                    float alpha = 0.0f; // Weight towards High index

                    // Case A: Extrapolation Low (Clamp to 0)
                    if (s_real <= sorted_centers[0].real()) {
                        idx_low = 0; 
                        idx_high = 0;
                        alpha = 0.0f;
                    } 
                    // Case B: Extrapolation High (Clamp to N-1)
                    else if (s_real >= sorted_centers[_nref_-1].real()) {
                        idx_low = _nref_-1; 
                        idx_high = _nref_-1;
                        alpha = 0.0f;
                    } 
                    // Case C: Interpolation
                    else {
                        // Scan for bracket [k, k+1]
                        for (int k = 0; k < _nref_ - 1; k++) {
                            float s1 = sorted_centers[k].real();
                            float s2 = sorted_centers[k+1].real();
                            if (s_real >= s1 && s_real <= s2) {
                                idx_low = k;
                                idx_high = k + 1;
                                // Linear Weight: (val - low) / (high - low)
                                alpha = (s_real - s1) / (s2 - s1);
                                break;
                            }
                        }
                    }

                    // Store Results in 4D arrays
                    ref_labels_low[iz][iw][iy][ix] = idx_low;
                    ref_labels_high[iz][iw][iy][ix] = idx_high;
                    ref_weights[iz][iw][iy][ix] = alpha;
                }
            }

            // =========================================================
            // 4. PADDING (Copy edge values to padded region)
            // =========================================================
            auto pad_edges = [&](auto& array) {
                // Y Padding
                for (int iy=_ny_; iy < _ny_ + pady; ++iy) {
                    for (int ix=0; ix < _nx_; ++ix) {
                        array[iz][iw][iy][ix] = array[iz][iw][_ny_-1][ix];
                    }
                }
                // X Padding
                for (int iy=0; iy < _ny_ + pady; ++iy) {
                    for (int ix=_nx_; ix < _nx_ + padx; ++ix) {
                        array[iz][iw][iy][ix] = array[iz][iw][iy][_nx_-1];
                    }
                }
            };

            pad_edges(ref_labels_low);
            pad_edges(ref_labels_high);
            pad_edges(ref_weights);
        }
    });
    is_sampled[iz] = true;
}

std::future<void> RefSampler::sample_at_depth_async(std::shared_ptr<complex4DReg> slow, size_t iz) {
	return std::async(std::launch::async, [this, slow, iz]() {
			sample_at_depth(slow, iz);
	});
}

