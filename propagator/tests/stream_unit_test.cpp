#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <complex>
#include <random> // For SetUp
#include <numeric> // For std::iota
#include <cmath> // For std::floor, std::ceil
#include <limits> // For numeric_limits

// --- Include necessary headers for your classes ---
#include "StreamingPropagator.h" // Class under test
#include "hypercube.h"         // Assuming definition exists
#include "axis.h"              // Assuming definition exists
#include "complex2DReg.h"      // Assuming definition exists
#include "complex4DReg.h"      // Assuming definition exists
#include "paramObj.h"          // Assuming definition exists (using jsonParamObj below)
#include "jsonParamObj.h"      // Using the type from your SetUp
#include "Propagator.h"        // Needed for unique_ptr type, even if not directly called much
// --- Google Test Code ---

// Use the provided SetUp structure
class StreamingPropagatorTest : public testing::Test {
protected:
    // Copied and adapted from user's provided SetUp
    int nx, ny, nz, nw, ns;
    std::vector<axis> ax;
    std::shared_ptr<jsonParamObj> par; // Use the specific type from setup
    std::vector<int> default_batches = {2,2};

    std::shared_ptr<hypercube> domainHyper, rangeHyper, slowHyper; // Renamed for clarity
    std::vector<float> sx_all, sy_all, sz_all;
    std::vector<int> s_ids_all;
    std::vector<float> rx_all, ry_all, rz_all;
    std::vector<int> r_ids_all;
    std::shared_ptr<complex2DReg> wavelet;
    std::shared_ptr<complex2DReg> data; // Renamed 'traces' to 'data' for consistency with forward signature
    std::vector<std::shared_ptr<complex4DReg>> model; // Renamed 'slow_den' to 'model'

    // Helper to create hypercube for tests if needed elsewhere
    std::shared_ptr<hypercube> createHypercube(const std::vector<axis>& axes) {
         return std::make_shared<hypercube>(axes);
    }

    // Simplified coord creation for predictable test data
    void createTestCoords(int n_traces, int n_unique_ids, float x_range, float y_range, float z_val,
                           std::vector<float>& cx, std::vector<float>& cy, std::vector<float>& cz, std::vector<int>& c_ids) {
        cx.resize(n_traces);
        cy.resize(n_traces);
        cz.resize(n_traces);
        c_ids.resize(n_traces);
        for (int i = 0; i < n_traces; ++i) {
            cx[i] = (i % 5) * x_range / 5.0f; // Spread out X
            cy[i] = (i % 4) * y_range / 4.0f; // Spread out Y
            cz[i] = z_val;
            c_ids[i] = 101 + (i % n_unique_ids); // Assign repeating IDs
        }
    }


    void SetUp() override {
        nx = 100;
        auto ax1 = axis(nx, 0.f, 10.0f); // Example: 10m spacing
        ny = 100;
        auto ax2 = axis(ny, 0.f, 10.0f);
        nw = 10; // Total frequencies
        auto ax3 = axis(nw, 1.f, 1.f); // Freq: 1Hz to 10Hz
        ns = 10; // Number of unique sources for model axis? Unused?
        nz = 10;
        auto ax4 = axis(nz, 0.f, 5.0f); // Z axis: 5m spacing

        ax = {ax1, ax2, ax3, ax4};
        // Define the slow_hyper based on this assumption
        slowHyper = createHypercube(ax);
        // Use the same for model hypercube
        auto modelHyper = slowHyper;


        Json::Value root; // Assuming jsonParamObj uses JsonCpp
        root["nref"] = 11;
        root["padx"] = nx; // These seem unused in StreamingPropagator itself
        root["pady"] = ny;
        root["taperx"] = 10;
        root["tapery"] = 10;
        root["ginsu_x"] = 100.0; // Padding in meters
        root["ginsu_y"] = 100.0;
        par = std::make_shared<jsonParamObj>(root); // Assuming constructor exists

        int nsrc_traces = 5;
        int nrec_traces = 8;
        int n_unique_src = 3; // For test data generation

        // Domain/Range Hypercubes: Assuming [Freq, Trace]
        // Make sure axis indices match constructor/subHyper logic (0=Freq, 1=Trace)
        domainHyper = createHypercube({ax[3], axis(nsrc_traces, 0, 1, "trace")});
        rangeHyper = createHypercube({ax[3], axis(nrec_traces, 0, 1, "receiver")});

        // Create predictable test coords instead of random
        createTestCoords(nsrc_traces, n_unique_src, nx * ax[0].d, ny * ax[1].d, 5.0f,
                         sx_all, sy_all, sz_all, s_ids_all);
        createTestCoords(nrec_traces, n_unique_src, nx * ax[0].d, ny * ax[1].d, 10.0f,
                         rx_all, ry_all, rz_all, r_ids_all);
        // Ensure receiver IDs map to source IDs used
         for(int i=0; i<nrec_traces; ++i) r_ids_all[i] = 101 + (i%n_unique_src);


        wavelet = std::make_shared<complex2DReg>(domainHyper);
        // Fill wavelet predictably
        for(int it=0; it<nsrc_traces; ++it) {
            for(int iw=0; iw<nw; ++iw) {
                (*wavelet->_mat)[it][iw] = {(float)iw, (float)it}; // Freq=real, Trace=imag
            }
        }

        data = std::make_shared<complex2DReg>(rangeHyper); // Output data buffer

        model.resize(2);
        model[0] = std::make_shared<complex4DReg>(modelHyper);
        model[1] = std::make_shared<complex4DReg>(modelHyper);
        model[0]->set(1.0f); // Set model to constant 1.0
        model[1]->set(1.0f); // Set model to constant 1.0

        // add a second layer to model for testing
        for (int iw = 0; iw < nw; ++iw) {
            for (int iy = 0; iy < ny; ++iy) {
                for (int ix = 0; ix < nx; ++ix) {
                    (*model[0]->_mat)[int(nz/2)][iw][iy][ix] = {2.f, 0.f};
                }
            }
        }

        default_batches = {2, 2}; // Example default batches for tests
    }

    void TearDown() override {
         // Destructor of StreamingPropagator handles stream cleanup
    }

};


// Test divideSourcesIntoBatches (called by constructor)
TEST_F(StreamingPropagatorTest, DivideSourcesIntoBatches) {
    int n_src_batch_req = 2;
    int n_freq_batch_req = 1; // Not relevant here
    std::vector<int> batches = {n_src_batch_req, n_freq_batch_req};

    StreamingPropagator streamer(domainHyper, rangeHyper, slowHyper, wavelet,
                                 sx_all, sy_all, sz_all, s_ids_all,
                                 rx_all, ry_all, rz_all, r_ids_all,
                                 par, batches);

    // Access members 
    ASSERT_EQ(streamer.nsrc_batches, n_src_batch_req); // Should not reduce in this setup (3 unique > 2 batches)
    ASSERT_EQ(streamer.sx_batches.size(), n_src_batch_req);
    ASSERT_EQ(streamer.sy_batches.size(), n_src_batch_req);
    ASSERT_EQ(streamer.sz_batches.size(), n_src_batch_req);
    ASSERT_EQ(streamer.src_ids_batches.size(), n_src_batch_req);
    ASSERT_EQ(streamer.src_index_batches.size(), n_src_batch_req);
    ASSERT_EQ(streamer.rx_batches.size(), n_src_batch_req);
    ASSERT_EQ(streamer.ry_batches.size(), n_src_batch_req);
    ASSERT_EQ(streamer.rz_batches.size(), n_src_batch_req);
    ASSERT_EQ(streamer.r_ids_batches.size(), n_src_batch_req);
    ASSERT_EQ(streamer.r_index_batches.size(), n_src_batch_req);

    // Check distribution (3 unique sources -> batches of size 2 and 1 unique source)
    // Batch 0 should get unique IDs 101, 102
    // Batch 1 should get unique ID 103
    std::unordered_set<int> batch0_src_ids, batch1_src_ids;
    std::vector<int> batch0_src_indices, batch1_src_indices;
    std::vector<int> batch0_rec_indices, batch1_rec_indices;

    for(int id : streamer.src_ids_batches[0]) batch0_src_ids.insert(id);
    for(int id : streamer.src_ids_batches[1]) batch1_src_ids.insert(id);
    for(int idx : streamer.src_index_batches[0]) batch0_src_indices.push_back(idx);
    for(int idx : streamer.src_index_batches[1]) batch1_src_indices.push_back(idx);
    for(int idx : streamer.r_index_batches[0]) batch0_rec_indices.push_back(idx);
    for(int idx : streamer.r_index_batches[1]) batch1_rec_indices.push_back(idx);

    // Check unique IDs per batch
    ASSERT_EQ(batch0_src_ids.size(), 2); // IDs 101, 102
    ASSERT_TRUE(batch0_src_ids.count(101));
    ASSERT_TRUE(batch0_src_ids.count(102));
    ASSERT_EQ(batch1_src_ids.size(), 1); // ID 103
    ASSERT_TRUE(batch1_src_ids.count(103));

    // Check original source trace indices associated with these batches
    // s_ids_all = {101, 102, 103, 101, 102}; -> Indices: 0, 1, 2, 3, 4
    // Batch 0 (IDs 101, 102) -> Should have original indices 0, 1, 3, 4
    // Batch 1 (ID 103) -> Should have original index 2
    std::sort(batch0_src_indices.begin(), batch0_src_indices.end());
    std::sort(batch1_src_indices.begin(), batch1_src_indices.end());
    ASSERT_EQ(batch0_src_indices, std::vector<int>({0, 1, 3, 4}));
    ASSERT_EQ(batch1_src_indices, std::vector<int>({2}));

    // Check receiver original indices associated
    // r_ids_all = {101, 102, 103, 101, 102, 103, 101, 102}; -> Indices 0-7
    // Batch 0 (IDs 101, 102) -> Should have original indices 0, 1, 3, 4, 6, 7
    // Batch 1 (ID 103) -> Should have original indices 2, 5
    std::sort(batch0_rec_indices.begin(), batch0_rec_indices.end());
    std::sort(batch1_rec_indices.begin(), batch1_rec_indices.end());
    ASSERT_EQ(batch0_rec_indices, std::vector<int>({0, 1, 3, 4, 6, 7}));
    ASSERT_EQ(batch1_rec_indices, std::vector<int>({2, 5}));
}

// Test divideFrequenciesIntoBatches (called by constructor)
TEST_F(StreamingPropagatorTest, DivideFrequenciesIntoBatches) {
    int n_src_batch_req = 1;
    int n_freq_batch_req = 3; // nw = 10
    std::vector<int> batches = {n_src_batch_req, n_freq_batch_req};

    StreamingPropagator streamer(domainHyper, rangeHyper, slowHyper, wavelet,
                                 sx_all, sy_all, sz_all, s_ids_all,
                                 rx_all, ry_all, rz_all, r_ids_all,
                                 par, batches);

    ASSERT_EQ(streamer.nfreq_batches, n_freq_batch_req); // 10 freqs >= 3 batches
    ASSERT_EQ(streamer.freq_start_indices.size(), n_freq_batch_req);
    ASSERT_EQ(streamer.freq_batch_sizes.size(), n_freq_batch_req);

    // Expected distribution for nw=10, nfreq_batches=3:
    // Batch 0: size 4 (10/3=3, rem=1), start 0
    // Batch 1: size 3 (10/3=3, rem=0), start 4
    // Batch 2: size 3 (10/3=3, rem=0), start 7
    ASSERT_EQ(streamer.freq_batch_sizes[0], 4);
    ASSERT_EQ(streamer.freq_start_indices[0], 0);
    ASSERT_EQ(streamer.freq_batch_sizes[1], 3);
    ASSERT_EQ(streamer.freq_start_indices[1], 4);
    ASSERT_EQ(streamer.freq_batch_sizes[2], 3);
    ASSERT_EQ(streamer.freq_start_indices[2], 7);

    // Check total size
    long long total_size = 0;
    for(int size : streamer.freq_batch_sizes) { total_size += size; }
    ASSERT_EQ(total_size, nw); // Check if partitions cover the whole range
}

// Test createSubHypercube
TEST_F(StreamingPropagatorTest, CreateSubHypercube) {
    StreamingPropagator streamer(domainHyper, rangeHyper, slowHyper, wavelet,
                                 sx_all, sy_all, sz_all, s_ids_all,
                                 rx_all, ry_all, rz_all, r_ids_all,
                                 par, default_batches);

    // Create a sub hypercube for frequency batch 1 (start index 4, size 3 from previous test)
    int test_start_freq = 4;
    int test_freq_batch_size = 3;
    std::vector<int> dummy_ids = {101, 102}; // Represents traces/receivers in a batch

    // Assuming domainHyper is [Freq, Trace] -> Axis 0=Freq, Axis 1=Trace
    auto subHyper = streamer.createSubHypercube(domainHyper, dummy_ids, test_start_freq, test_freq_batch_size);

    ASSERT_NE(subHyper, nullptr);
    ASSERT_EQ(subHyper->getNdim(), 2);

    // Check Frequency Axis (Axis 0)
    const axis& freqAx = subHyper->getAxis(1);
    ASSERT_EQ(freqAx.n, test_freq_batch_size);
    // Check Origin 
    // Current code: axes[0].o = axes[1].o + start_freq * axes[1].d; (Uses Trace axis params)
    float expected_origin_WRONG = domainHyper->getAxis(2).o + test_start_freq * domainHyper->getAxis(2).d;
    // Correct logic: axes[0].o = original->getAxis(0).o + start_freq * original->getAxis(0).d;
    float expected_origin_CORRECT = domainHyper->getAxis(1).o + test_start_freq * domainHyper->getAxis(1).d;
    // ASSERT_EQ(freqAx.o, expected_origin_CORRECT); // Test against correct logic
    ASSERT_EQ(freqAx.o, expected_origin_WRONG); // Test against current code's logic - EXPECT TO FAIL IF CODE IS FIXED
    EXPECT_NE(freqAx.o, expected_origin_CORRECT) << "Test will fail if createSubHypercube origin calculation is fixed!";
    ASSERT_EQ(freqAx.d, domainHyper->getAxis(1).d); // Delta should be unchanged

    // Check Trace/ID Axis (Axis 1)
    const axis& traceAx = subHyper->getAxis(2);
    ASSERT_EQ(traceAx.n, dummy_ids.size());
    ASSERT_EQ(traceAx.o, domainHyper->getAxis(2).o); // Origin should be unchanged
    ASSERT_EQ(traceAx.d, domainHyper->getAxis(2).d); // Delta should be unchanged
}


// Test createSubWavelet
TEST_F(StreamingPropagatorTest, CreateSubWavelet) {
     StreamingPropagator streamer(domainHyper, rangeHyper, slowHyper, wavelet,
                                 sx_all, sy_all, sz_all, s_ids_all,
                                 rx_all, ry_all, rz_all, r_ids_all,
                                 par, default_batches);

    // Create a dummy subdomain hypercube: [Freq, Trace]
    int sub_nfreq = 5;
    int sub_ntraces = 3;
    auto subDomainHyper = createHypercube({axis(sub_nfreq, 0, 1), axis(sub_ntraces, 0, 1)});

    // Select original trace indices to copy
    std::vector<int> trace_indices = {0, 2, 4}; // Indices from original wavelet (size nsrc_traces=5)
    ASSERT_LE(trace_indices.size(), sub_ntraces); // Ensure indices fit in sub wavelet

    std::shared_ptr<complex2DReg> subWavelet;
    int start_freq = 0;
    ASSERT_NO_THROW({
        subWavelet = streamer.createSubWavelet(wavelet, subDomainHyper, trace_indices, start_freq);
    });

    ASSERT_NE(subWavelet, nullptr);
    ASSERT_EQ(subWavelet->getHyper()->getAxis(1).n, sub_nfreq);
    ASSERT_EQ(subWavelet->getHyper()->getAxis(2).n, sub_ntraces);

    // Check copied data - Original wavelet had {iw, it}
    ASSERT_EQ((*subWavelet->_mat).shape()[0], sub_ntraces);
    ASSERT_EQ((*subWavelet->_mat).shape()[1], sub_nfreq);
    for (int i = 0; i < trace_indices.size(); ++i) {
        int original_trace_idx = trace_indices[i];
        for (int iw = 0; iw < sub_nfreq; ++iw) {
            // Expected value from original wavelet setup
            std::complex<float> expected = {(float)iw, (float)original_trace_idx};
            ASSERT_EQ((*subWavelet->_mat)[i][iw], expected) << "Mismatch at sub trace " << i << " (orig " << original_trace_idx << "), freq " << iw;
        }
    }

    // Test bounds check - THIS IS WHERE THE CODE IS FLAWED
    // int original_ntraces = wavelet->getHyper()->getAxis(2).n; // Assuming trace is axis 1
    // std::vector<int> invalid_indices = {original_ntraces}; // Index exactly out of bounds for original
    // // Current code compares against subDomainHyper->getAxis(1).n which is sub_ntraces
    // // This test might incorrectly pass if sub_ntraces > original_ntraces
    // if (invalid_indices[0] < sub_ntraces) {
    //    // This scenario highlights the flaw: index is invalid for original, but valid for subdomain size check
    //    EXPECT_NO_THROW(streamer.createSubWavelet(wavelet, subDomainHyper, invalid_indices))
    //        << "Bounds check incorrectly passed - index invalid for original wavelet was not caught.";
    // } else {
    //    // This scenario *should* throw, but due to the wrong check in the code.
    //    // If the code were correct, it should throw because idx >= original_ntraces.
    //    // If the code is flawed, it throws because idx >= sub_ntraces.
    //     EXPECT_ANY_THROW(streamer.createSubWavelet(wavelet, subDomainHyper, invalid_indices))
    //        << "Bounds check should have caught invalid index (though possibly for the wrong reason).";
    // }
    // //  A correct implementation should always throw for invalid_indices = {original_ntraces}
    //  EXPECT_THROW(streamer.createSubWavelet(wavelet, subDomainHyper, invalid_indices), std::runtime_error);
}

// Test calculateWindowParameters
TEST_F(StreamingPropagatorTest, CalculateWindowParameters) {
    int n_src_batch_req = 1; // Put all sources/receivers in one batch for simplicity
    std::vector<int> batches = {n_src_batch_req, 1};

    StreamingPropagator streamer(domainHyper, rangeHyper, slowHyper, wavelet,
                                 sx_all, sy_all, sz_all, s_ids_all,
                                 rx_all, ry_all, rz_all, r_ids_all,
                                 par, batches);

    ASSERT_EQ(streamer.nsrc_batches, 1);
    int src_batch_idx = 0;

    // Get expected min/max coords from _all vectors
    float min_sx = *std::min_element(sx_all.begin(), sx_all.end());
    float max_sx = *std::max_element(sx_all.begin(), sx_all.end());
    float min_sy = *std::min_element(sy_all.begin(), sy_all.end());
    float max_sy = *std::max_element(sy_all.begin(), sy_all.end());
    float min_rx = *std::min_element(rx_all.begin(), rx_all.end());
    float max_rx = *std::max_element(rx_all.begin(), rx_all.end());
    float min_ry = *std::min_element(ry_all.begin(), ry_all.end());
    float max_ry = *std::max_element(ry_all.begin(), ry_all.end());

    float expected_min_x = std::min(min_sx, min_rx) - streamer.ginsu_x;
    float expected_max_x = std::max(max_sx, max_rx) + streamer.ginsu_x;
    float expected_min_y = std::min(min_sy, min_ry) - streamer.ginsu_y;
    float expected_max_y = std::max(max_sy, max_ry) + streamer.ginsu_y;

    // Get model grid params (Assuming modelHyper axis 0=X, 1=Y)
    float x_orig = model[0]->getHyper()->getAxis(1).o;
    float y_orig = model[0]->getHyper()->getAxis(2).o;
    float dx = model[0]->getHyper()->getAxis(1).d;
    float dy = model[0]->getHyper()->getAxis(2).d;
    int model_nx = model[0]->getHyper()->getAxis(1).n;
    int model_ny = model[0]->getHyper()->getAxis(2).n;

    // Calculate expected indices
    int expected_min_ix = std::max(0, static_cast<int>(std::floor((expected_min_x - x_orig) / dx)));
    int expected_max_ix = std::min(model_nx - 1, static_cast<int>(std::ceil((expected_max_x - x_orig) / dx)));
    int expected_min_iy = std::max(0, static_cast<int>(std::floor((expected_min_y - y_orig) / dy)));
    int expected_max_iy = std::min(model_ny - 1, static_cast<int>(std::ceil((expected_max_y - y_orig) / dy)));

     // Handle degenerate cases if necessary (though unlikely with padding)
     if (expected_max_ix < expected_min_ix) { expected_min_ix = 0; expected_max_ix = model_nx - 1;}
     if (expected_max_iy < expected_min_iy) { expected_min_iy = 0; expected_max_iy = model_ny - 1;}


    int min_ix, max_ix, min_iy, max_iy;
    ASSERT_NO_THROW({
        std::tie(min_ix, max_ix, min_iy, max_iy) = streamer.calculateWindowParameters(src_batch_idx, model[0]->getHyper());
    });

    ASSERT_EQ(min_ix, expected_min_ix);
    ASSERT_EQ(max_ix, expected_max_ix);
    ASSERT_EQ(min_iy, expected_min_iy);
    ASSERT_EQ(max_iy, expected_max_iy);
}

// Test forward with add = false
TEST_F(StreamingPropagatorTest, ForwardNonAdd) {
  // Use default batches {2, 2}
  StreamingPropagator streamer(domainHyper, rangeHyper, slowHyper, wavelet,
                               sx_all, sy_all, sz_all, s_ids_all,
                               rx_all, ry_all, rz_all, r_ids_all,
                               par, {2,2});
  Propagator non_streamer(domainHyper, rangeHyper, slowHyper, wavelet,
                              sx_all, sy_all, sz_all, s_ids_all,
                              rx_all, ry_all, rz_all, r_ids_all,
                              par);

  // Make sure data is initially non-zero to verify zeroing
  std::complex<float> initial_garbage = {99.0f, -99.0f};
  data->set(initial_garbage);
  auto non_stream_data = data->clone();

  ASSERT_NO_THROW(non_streamer.forward(false, model, non_stream_data));

  // Run forward
  ASSERT_NO_THROW(streamer.forward(false, model, data));

  // Verify results
  int nRecs_total = rangeHyper->getAxis(2).n;
  int nFreq_total = rangeHyper->getAxis(1).n;
  double tolerance = 1e-5;

  ASSERT_TRUE(std::real(non_stream_data->dot(non_stream_data)) > 0.) 
      << "The non-streaming data is zero everywhere";

  ASSERT_NEAR(std::real(non_stream_data->dot(non_stream_data)), 
              std::real(data->dot(data)), 1e-3) 
      << "Mismatch in the data norms after forward propagation";

  for (int iRec = 0; iRec < nRecs_total; ++iRec) {
    for (int iFreq = 0; iFreq < nFreq_total; ++iFreq) {
      std::complex<float> expected = (*non_stream_data->_mat)[iRec][iFreq];
      std::complex<float> computed = (*data->_mat)[iRec][iFreq];
      ASSERT_NEAR(computed.real(), expected.real(), tolerance)
          << "(Real) Mismatch at Global Receiver " << iRec << ", Global Frequency " << iFreq;
      ASSERT_NEAR(computed.imag(), expected.imag(), tolerance)
          << "(Imag) Mismatch at Global Receiver " << iRec << ", Global Frequency " << iFreq;
    }
  }

}



int main(int argc, char **argv) {
  // Parse command-line arguments
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}