  #pragma once
  #include "complex4DReg.h" 
  #include "float2DReg.h"
  #include <tbb/blocked_range.h>
  #include <tbb/parallel_for.h>
  #include <vector>
  #include <cmath>

  namespace SEP {

  // double PI = 4.*std::atan(1); // Usually better to define inside or use M_PI

  class Interpolation4D {

  public:
    Interpolation4D(std::shared_ptr<complex4DReg> model, std::shared_ptr<complex4DReg> data) {
      for (int i=0; i<4; ++i) { // Now loops 0..3
        ax_d.push_back(data->getHyper()->getAxis(i+1));
        ax_m.push_back(model->getHyper()->getAxis(i+1));
      }
    }

    void forward(bool add, std::shared_ptr<complex4DReg> model, std::shared_ptr<complex4DReg> data) {
      if (!add) data->scale(0);

      long long n1_d = ax_d[0].n;
      long long n2_d = ax_d[1].n;
      long long n3_d = ax_d[2].n;
      // long long n4_d = ax_d[3].n; // Used in parallel_for range

      long long n1_m = ax_m[0].n;
      long long n2_m = ax_m[1].n;
      long long n3_m = ax_m[2].n;
      long long n4_m = ax_m[3].n;

      // Process slowest Output axis (i4) in parallel
      tbb::parallel_for(tbb::blocked_range<int>(0, ax_d[3].n),
      [&](const tbb::blocked_range<int> &r4) {
        
        // Temporary buffers: Size is (Output Fast Axis) * (Model 2nd Axis)
        // We reuse this buffer for every i3 slice
        std::vector<std::complex<float>> temp_line1(n1_d * n2_m, std::complex<float>(0,0));

        for (int i4 = r4.begin(); i4 != r4.end(); i4++) {
          
          // Iterate over the next output axis (i3) serially within the thread
          for (int i3 = 0; i3 < n3_d; i3++) {

              // Clear temp buffer for this specific (i4, i3) output slice
              std::fill(temp_line1.begin(), temp_line1.end(), std::complex<float>(0,0));

              // Accumulate contributions from Model (j4, j3)
              for (int j4 = 0; j4 < n4_m; j4++) {
                  float filt4 = (*filter4)[i4][j4];
                  if (filt4 == 0) continue;

                  for (int j3 = 0; j3 < n3_m; j3++) {
                      float filt3 = (*filter3)[i3][j3];
                      if (filt3 == 0) continue;

                      float w_34 = filt4 * filt3;

                      // --- Core 2D Interpolation Logic (same as 3D) ---
                      // Step 1: Filter along Axis 1 (Model j1 -> Data i1)
                      for (int j2 = 0; j2 < n2_m; j2++) {
                          for (int i1 = 0; i1 < n1_d; i1++) {
                              std::complex<float> sum(0.0, 0.0);
                              for (int j1 = 0; j1 < n1_m; j1++) {
                                  float filt1 = (*filter1)[i1][j1];
                                  if (filt1 != 0) {
                                      // Calculate 4D Index for Model
                                      long long ind_m = j1 + j2*n1_m + j3*n1_m*n2_m + j4*n1_m*n2_m*n3_m;
                                      sum += model->_mat->data()[ind_m] * filt1;
                                  }
                              }
                              // Accumulate into temp buffer
                              int ind_t1 = i1 + j2*n1_d;
                              temp_line1[ind_t1] += sum * w_34;
                          }
                      }
                  }
              }

              // Step 2: Filter along Axis 2 using temp_line1 -> Write to Data
              for (int i2 = 0; i2 < n2_d; i2++) {
                  for (int i1 = 0; i1 < n1_d; i1++) {
                      std::complex<float> sum(0.0, 0.0);
                      for (int j2 = 0; j2 < n2_m; j2++) {
                          float filt2 = (*filter2)[i2][j2];
                          if (filt2 != 0) {
                              int ind_t1 = i1 + j2*n1_d;
                              sum += temp_line1[ind_t1] * filt2;
                          }
                      }
                      // Final 4D index calculation
                      long long ind_d = i1 + i2*n1_d + i3*n1_d*n2_d + i4*n1_d*n2_d*n3_d;
                      data->getVals()[ind_d] += sum;
                  }
              }
          }
        }
      });
    }

    void adjoint(bool add, std::shared_ptr<complex4DReg> model, std::shared_ptr<complex4DReg> data) {
      if (!add) model->scale(0);

      long long n1_d = ax_d[0].n;
      long long n2_d = ax_d[1].n;
      long long n3_d = ax_d[2].n;
      long long n4_d = ax_d[3].n;

      long long n1_m = ax_m[0].n;
      long long n2_m = ax_m[1].n;
      long long n3_m = ax_m[2].n;
      // long long n4_m = ax_m[3].n;

      // Process slowest Model axis (j4) in parallel
      tbb::parallel_for(tbb::blocked_range<int>(0, ax_m[3].n),
      [&](const tbb::blocked_range<int> &r4) {
        
        std::vector<std::complex<float>> temp_line1(n1_d * n2_m, std::complex<float>(0,0));

        for (int j4 = r4.begin(); j4 != r4.end(); j4++) {
          
          for (int j3 = 0; j3 < n3_m; j3++) {
              
              // Clear buffer for this (j4, j3) model slice
              std::fill(temp_line1.begin(), temp_line1.end(), std::complex<float>(0,0));

              // Loop over all Data slices (i4, i3) that contribute to this model slice
              for (int i4 = 0; i4 < n4_d; i4++) {
                  float filt4 = (*filter4)[i4][j4];
                  if (filt4 == 0) continue;

                  for (int i3 = 0; i3 < n3_d; i3++) {
                      float filt3 = (*filter3)[i3][j3];
                      if (filt3 == 0) continue;
                      
                      float w_34 = filt4 * filt3;

                      // Step 1 & 2: Accumulate Data * F3 * F4 * F2 into temp_line1
                      for (int i2 = 0; i2 < n2_d; i2++) {
                          for (int j2 = 0; j2 < n2_m; j2++) {
                              float filt2 = (*filter2)[i2][j2];
                              if (filt2 == 0) continue;

                              for (int i1 = 0; i1 < n1_d; i1++) {
                                  long long ind_d = i1 + i2*n1_d + i3*n1_d*n2_d + i4*n1_d*n2_d*n3_d;
                                  int ind_t1 = i1 + j2*n1_d;
                                  
                                  temp_line1[ind_t1] += data->_mat->data()[ind_d] * w_34 * filt2;
                              }
                          }
                      }
                  }
              }

              // Step 3: Apply filter1 and write to Model
              for (int j2 = 0; j2 < n2_m; j2++) {
                  for (int j1 = 0; j1 < n1_m; j1++) {
                      std::complex<float> sum(0.0, 0.0);
                      for (int i1 = 0; i1 < n1_d; i1++) {
                          float filt1 = (*filter1)[i1][j1];
                          if (filt1 != 0) {
                              int ind_t1 = i1 + j2*n1_d;
                              sum += temp_line1[ind_t1] * filt1;
                          }
                      }
                      long long ind_m = j1 + j2*n1_m + j3*n1_m*n2_m + j4*n1_m*n2_m*n3_m;
                      model->getVals()[ind_m] += sum;
                  }
              }
          }
        }
      });
    }

  protected:
    std::vector<axis> ax_d, ax_m;
    std::shared_ptr<float2D> filter1, filter2, filter3, filter4; // Added filter4
  };

  }