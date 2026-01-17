#pragma once
#include "complex4DReg.h"
#include "float2DReg.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/cache_aligned_allocator.h> 
#include <vector>
#include <cmath>
#include <algorithm>

namespace SEP {

struct SparseFilter {
    std::shared_ptr<float2D> weights;
    std::vector<int> start;      // Forward: For output i, loop j from start[i] to end[i]
    std::vector<int> end;
    std::vector<int> start_inv;  // Adjoint: For input j, loop i from start_inv[j] to end_inv[j]
    std::vector<int> end_inv;
    
    // Helper to get raw data pointer
    float* data() const { return weights->data(); }
};

class Spline4D {

public:
    Spline4D(std::shared_ptr<complex4DReg> model, std::shared_ptr<complex4DReg> data, float a, float b, std::vector<float> taper_perc) {
        if (taper_perc.size() < 4) throw std::runtime_error("Spline4D requires 4 taper percentages");

		for (int i=0; i<4; ++i) { // Now loops 0..3
			ax_d.push_back(data->getHyper()->getAxis(i+1));
			ax_m.push_back(model->getHyper()->getAxis(i+1));
		}

        f4 = buildFilter(ax_d[3], ax_m[3], a, b, taper_perc[3]);
        f3 = buildFilter(ax_d[2], ax_m[2], a, b, taper_perc[2]);
        f2 = buildFilter(ax_d[1], ax_m[1], a, b, taper_perc[1]);
        f1 = buildFilter(ax_d[0], ax_m[0], a, b, taper_perc[0]);
    }

    void forward(bool add, std::shared_ptr<complex4DReg> model, std::shared_ptr<complex4DReg> data) {
        if (!add) data->scale(0);

        const long long n1_d = ax_d[0].n;
        const long long n2_m = ax_m[1].n;
        const long long n3_d = ax_d[2].n;
        
        // Raw pointers to Filter Weights (Hoist shared_ptr dereference out of loops)
        const float* __restrict__ w1 = f1.data();
        const float* __restrict__ w2 = f2.data();
        const float* __restrict__ w3 = f3.data();
        const float* __restrict__ w4 = f4.data();

        std::complex<float>* __restrict__ p_data = data->getVals();
        const std::complex<float>* __restrict__ p_model = model->getVals();

        // Dimensions for pointer arithmetic
        const long long str_m2 = ax_m[0].n;
        const long long str_m3 = str_m2 * ax_m[1].n;
        const long long str_m4 = str_m3 * ax_m[2].n;

        const long long str_d2 = ax_d[0].n;
        const long long str_d3 = str_d2 * ax_d[1].n;
        const long long str_d4 = str_d3 * ax_d[2].n;

        tbb::parallel_for(tbb::blocked_range<int>(0, ax_d[3].n),
        [&](const tbb::blocked_range<int> &r4) {
            
            // Thread-local buffer (std::vector allocation is fast enough here)
            std::vector<std::complex<float>, tbb::cache_aligned_allocator<std::complex<float>>> temp_line1(n1_d * n2_m);

            for (int i4 = r4.begin(); i4 != r4.end(); i4++) {
                
                // Get row of filter 4
                const float* w4_row = w4 + i4 * ax_m[3].n;

                for (int i3 = 0; i3 < n3_d; i3++) {
                    
                    std::fill(temp_line1.begin(), temp_line1.end(), std::complex<float>(0,0));
                    
                    // Get row of filter 3
                    const float* w3_row = w3 + i3 * ax_m[2].n;

                    // 1. Model -> Temp
                    for (int j4 = f4.start[i4]; j4 < f4.end[i4]; j4++) {
                        const float val4 = w4_row[j4];

                        for (int j3 = f3.start[i3]; j3 < f3.end[i3]; j3++) {
                            const float val34 = val4 * w3_row[j3];
                            
                            // Pointer to Model start for (j3, j4)
                            const std::complex<float>* pm_base = p_model + j4 * str_m4 + j3 * str_m3;

                            for (int j2 = 0; j2 < n2_m; j2++) {
                                const std::complex<float>* pm = pm_base + j2 * str_m2;
                                
                                for (int i1 = 0; i1 < n1_d; i1++) {
                                    std::complex<float> sum(0.0f, 0.0f);
                                    
                                    // Pointer to specific row in Filter 1
                                    const float* w1_row = w1 + i1 * ax_m[0].n;
                                    
                                    #pragma GCC ivdep
                                    for (int j1 = f1.start[i1]; j1 < f1.end[i1]; j1++) {
                                        sum += pm[j1] * w1_row[j1];
                                    }
                                    temp_line1[i1 + j2 * n1_d] += sum * val34;
                                }
                            }
                        }
                    }

                    // 2. Temp -> Data
                    std::complex<float>* pd_base = p_data + i4 * str_d4 + i3 * str_d3;
                    
                    for (int i2 = 0; i2 < ax_d[1].n; i2++) {
                        std::complex<float>* pd = pd_base + i2 * str_d2;
                        const float* w2_row = w2 + i2 * ax_m[1].n;

                        for (int i1 = 0; i1 < n1_d; i1++) {
                            std::complex<float> sum(0.0f, 0.0f);
                            
                            #pragma GCC ivdep
                            for (int j2 = f2.start[i2]; j2 < f2.end[i2]; j2++) {
                                sum += temp_line1[i1 + j2 * n1_d] * w2_row[j2];
                            }
                            pd[i1] += sum;
                        }
                    }
                }
            }
        });
    }

    void adjoint(bool add, std::shared_ptr<complex4DReg> model, std::shared_ptr<complex4DReg> data) {
        if (!add) model->scale(0);

        const long long n1_d = ax_d[0].n;
        const long long n2_m = ax_m[1].n;
        const long long n3_m = ax_m[2].n;

        // Raw pointers
        const float* __restrict__ w1 = f1.data();
        const float* __restrict__ w2 = f2.data();
        const float* __restrict__ w3 = f3.data();
        const float* __restrict__ w4 = f4.data();

        std::complex<float>* __restrict__ p_model = model->getVals();
        const std::complex<float>* __restrict__ p_data = data->getVals();

        // Strides
        const long long str_m2 = ax_m[0].n;
        const long long str_m3 = str_m2 * ax_m[1].n;
        const long long str_m4 = str_m3 * ax_m[2].n;

        const long long str_d2 = ax_d[0].n;
        const long long str_d3 = str_d2 * ax_d[1].n;
        const long long str_d4 = str_d3 * ax_d[2].n;

        tbb::parallel_for(tbb::blocked_range<int>(0, ax_m[3].n),
        [&](const tbb::blocked_range<int> &r4) {
            
            std::vector<std::complex<float>, tbb::cache_aligned_allocator<std::complex<float>>> temp_line1(n1_d * n2_m);

            for (int j4 = r4.begin(); j4 != r4.end(); j4++) {
                for (int j3 = 0; j3 < n3_m; j3++) {
                    
                    std::fill(temp_line1.begin(), temp_line1.end(), std::complex<float>(0,0));

                    // 1. Data -> Temp (Using Inverse Bounds for j4 and j3)
                    // We iterate i4/i3 only where they impact j4/j3
                    for (int i4 = f4.start_inv[j4]; i4 < f4.end_inv[j4]; i4++) {
                        const float val4 = w4[i4 * ax_m[3].n + j4]; // Access [i][j]

                        for (int i3 = f3.start_inv[j3]; i3 < f3.end_inv[j3]; i3++) {
                            const float val34 = val4 * w3[i3 * ax_m[2].n + j3];

                            const std::complex<float>* pd_base = p_data + i4 * str_d4 + i3 * str_d3;

                            for (int i2 = 0; i2 < ax_d[1].n; i2++) {
                                const std::complex<float>* pd = pd_base + i2 * str_d2;
                                
                                // Sparse F2 Adjoint: loop j2, scatter i2 to it
                                // Actually, simpler: Loop i2 (dense), and for each i2, accumulate to valid j2s?
                                // No, here we are reading Data(i2) and writing to Temp(j2).
                                // Temp is sized [n1_d * n2_m]. 
                                // We iterate Dense i2.
                                const float* w2_row = w2 + i2 * ax_m[1].n;
                                
                                for (int j2 = f2.start[i2]; j2 < f2.end[i2]; j2++) {
                                    float w_total = val34 * w2_row[j2];
                                    
                                    // Dense vector addition
                                    #pragma GCC ivdep
                                    for (int i1 = 0; i1 < n1_d; i1++) {
                                        temp_line1[i1 + j2 * n1_d] += pd[i1] * w_total;
                                    }
                                }
                            }
                        }
                    }

                    // 2. Temp -> Model (Using Inverse Bounds for j1)
                    std::complex<float>* pm_base = p_model + j4 * str_m4 + j3 * str_m3;

                    for (int j2 = 0; j2 < n2_m; j2++) {
                        std::complex<float>* pm = pm_base + j2 * str_m2;
                        
                        for (int j1 = 0; j1 < ax_m[0].n; j1++) {
                            std::complex<float> sum(0.0f, 0.0f);
                            
                            // CRITICAL OPTIMIZATION: 
                            // Loop only over i1 that contribute to this j1
                            for (int i1 = f1.start_inv[j1]; i1 < f1.end_inv[j1]; i1++) {
                                // Weight access: weights[i1][j1]
                                sum += temp_line1[i1 + j2 * n1_d] * w1[i1 * ax_m[0].n + j1];
                            }
                            
                            pm[j1] += sum;
                        }
                    }
                }
            }
        });
    }

private:
    SparseFilter f1, f2, f3, f4;
	std::vector<axis> ax_d, ax_m;
	
    SparseFilter buildFilter(axis ax_d, axis ax_m, float a, float b, float tap_perc) {
        SparseFilter res;
        res.start.resize(ax_d.n);
        res.end.resize(ax_d.n);
        res.start_inv.assign(ax_m.n, ax_d.n); // Default min = max_dim
        res.end_inv.assign(ax_m.n, 0);        // Default max = 0
        
        res.weights = std::make_shared<float2D>(boost::extents[ax_d.n][ax_m.n]);
        std::fill(res.weights->data(), res.weights->data() + res.weights->num_elements(), 0.0f);

        const float new_d = (1.0f + tap_perc) * ax_m.d;

        // Step 1: Compute spline weights
        for (int i = 0; i < ax_d.n; ++i) {
            for (int j = 0; j < ax_m.n; ++j) {
                const float x = std::abs(i * ax_d.d - j * ax_m.d);
                const float y = x / new_d;
                
                if (y < 1.0f) {
                    const float y2 = y * y;
                    const float y3 = y2 * y;
                    (*res.weights)[i][j] = ((-6*a - 9*b + 12) * y3 + 
                                            (6*a + 12*b - 18) * y2 - 
                                            2*b + 6) / 6.0f;
                } else if (y < 2.0f) {
                    const float y2 = y * y;
                    const float y3 = y2 * y;
                    (*res.weights)[i][j] = ((-6*a - b) * y3 + 
                                            (30*a + 6*b) * y2 + 
                                            (-48*a - 12*b) * y + 
                                            24*a + 8*b) / 6.0f;
                }
            }
        }

        // Step 2: Apply taper to edges
        for (int i = 0; i < ax_d.n; ++i) {
            // Your original loop range: j=1 to ax_m.n-1
            for (int j = 1; j < ax_m.n - 1; ++j) {
                float f = 0.0f;
                
                // Logic based on SUM (xx), not difference
                // Note: Your original code ignored 'o' here, so I will too to match exact behavior.
                float xx = (i * ax_d.d) + (j * ax_m.d); 
                float y = std::abs(xx) / new_d;

                if (y < 1.0f) {
                     float y2 = y * y;
                     float y3 = y2 * y;
                     f = ((-6*a - 9*b + 12) * y3 + (6*a + 12*b - 18) * y2 - 2*b + 6) / 6.0f;
                } else if (1.0f <= y && y < 2.0f) {
                     float y2 = y * y;
                     float y3 = y2 * y;
                     f = ((-6*a - b) * y3 + (30*a + 6*b) * y2 + (-48*a - 12*b) * y + 24*a + 8*b) / 6.0f;
                }

                // Add to Left Boundary
                (*res.weights)[i][j] += f;
                
                // Add to Right Boundary (Symmetric)
                // Ensure indices are safe
                int i_r = ax_d.n - i - 1;
                int j_r = ax_m.n - j - 1;
                if (i_r >= 0 && i_r < ax_d.n && j_r >= 0 && j_r < ax_m.n) {
                    (*res.weights)[i_r][j_r] += f;
                }
            }
		}

		// Step 3: Normalize each output column to sum to 1
        std::vector<float> col_norm(ax_m.n, 0.0f);
        
        // Accumulate Sum of Squares per Column (j)
        for (int i = 0; i < ax_d.n; ++i) {
            for (int j = 0; j < ax_m.n; ++j) {
                float val = (*res.weights)[i][j];
                col_norm[j] += val * val;
            }
        }

        // Apply Inverse Sqrt Norm
        for (int j = 0; j < ax_m.n; ++j) {
            if (col_norm[j] > 1e-20f) {
                col_norm[j] = 1.0f / std::sqrt(col_norm[j]);
            } else {
                col_norm[j] = 0.0f;
            }
        }
		
		// Normalize Weights
		for (int i = 0; i < ax_d.n; ++i) {
            for (int j = 0; j < ax_m.n; ++j) {
                (*res.weights)[i][j] *= col_norm[j];
            }
        }

        // Compute Forward and Inverse Bounds
        constexpr float THRESHOLD = 1e-8f;
        
        for (int i = 0; i < ax_d.n; ++i) {
            int min_j = ax_m.n;
            int max_j = -1;
            
            for (int j = 0; j < ax_m.n; ++j) {
                if (std::abs((*res.weights)[i][j]) > THRESHOLD) {
                    // Forward Bounds
                    if (j < min_j) min_j = j;
                    if (j > max_j) max_j = j;

                    // Inverse Bounds (Adjoint)
                    if (i < res.start_inv[j]) res.start_inv[j] = i;
                    if (i > res.end_inv[j])   res.end_inv[j]   = i;
                }
            }
            if (max_j == -1) { res.start[i] = 0; res.end[i] = 0; }
            else             { res.start[i] = min_j; res.end[i] = max_j + 1; }
        }

        // Finalize exclusive end for Inverse
        for(int j=0; j<ax_m.n; ++j) {
            if (res.end_inv[j] < res.start_inv[j]) { // Empty column
                res.start_inv[j] = 0;
                res.end_inv[j] = 0;
            } else {
                res.end_inv[j] += 1; // Make exclusive
            }
        }

        return res;
    }
};

}