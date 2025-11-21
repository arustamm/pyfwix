#include <memory>
#include <vector>
#include <complex>
#include <random> // For SetUp (kept for context, not used in main directly)
#include <numeric> // For std::iota (kept for context)
#include <cmath> // For std::floor, std::ceil (kept for context)
#include <limits> // For numeric_limits (kept for context)
#include <iostream> // For std::cout, std::cerr
#include <string>   // For std::stoi
#include <stdexcept> // For std::invalid_argument, std::out_of_range

// --- Include necessary headers for your classes ---
#include "StreamingPropagator.h" // Class under test
#include "hypercube.h"         // Assuming definition exists
#include "axis.h"              // Assuming definition exists
#include "complex2DReg.h"      // Assuming definition exists
#include "complex4DReg.h"      // Assuming definition exists
#include "paramObj.h"          // Assuming definition exists (using jsonParamObj below)
#include "jsonParamObj.h"      // Using the type from your SetUp
#include "Propagator.h"        // Needed for unique_ptr type (kept for context)

// --- NVTX Header for Profiling Ranges ---
#include <nvtx3/nvToolsExt.h> // Use nvtx3 namespace if available, else <nvToolsExt.h>

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

// Helper to create hypercube for tests if needed elsewhere
std::shared_ptr<hypercube> createHypercube(const std::vector<axis>& axes) {
    return std::make_shared<hypercube>(axes);
}

int main(int argc, char **argv) {

    // --- Command Line Argument Parsing for Batch Sizes ---
    std::vector<int> batches;
    std::vector<int> default_batches = {1, 1}; // Default values if not provided
    batches.push_back(1);
    int look_ahead = 1;

    if (argc == 1) {
        batches = default_batches;
        std::cout << "Using default batch sizes: [" << batches[0] << ", " << batches[1] << "]" << std::endl;
    } else if (argc == 3) {
        try {
            batches.push_back(std::stoi(argv[1]));
            look_ahead = std::stoi(argv[2]);
            std::cout << "Using command-line batch sizes: [" << batches[0] << ", " << batches[1] << "]" << ", look ahead: " << look_ahead << std::endl;
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Error: Invalid argument. Batch sizes must be integers." << std::endl;
            std::cerr << "Usage: " << argv[0] << " [batch_size_1 batch_size_2]" << std::endl;
            return 1;
        } catch (const std::out_of_range& oor) {
            std::cerr << "Error: Batch size argument out of range for integer." << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Error: Incorrect number of arguments." << std::endl;
        std::cerr << "Usage: " << argv[0] << " [batch_size_1 batch_size_2]" << std::endl;
        std::cerr << "Or run with no arguments to use default batches ["
                  << default_batches[0] << ", " << default_batches[1] << "]." << std::endl;
        return 1;
    }
    // Ensure batches have non-negative values if required by your propagator logic
     if (batches[0] <= 0 || batches[1] <= 0) {
         std::cerr << "Error: Batch sizes must be positive integers." << std::endl;
         return 1;
     }

    // --- Setup Code (largely unchanged) ---
    int nx, ny, nz, nw, ns;
    std::vector<axis> ax;
    std::shared_ptr<jsonParamObj> par; // Use the specific type from setup

    std::shared_ptr<hypercube> domainHyper, rangeHyper, slowHyper; // Renamed for clarity
    std::vector<float> sx_all, sy_all, sz_all;
    std::vector<int> s_ids_all;
    std::vector<float> rx_all, ry_all, rz_all;
    std::vector<int> r_ids_all;
    std::shared_ptr<complex2DReg> wavelet;
    std::shared_ptr<complex2DReg> data; // Renamed 'traces' to 'data' for consistency with forward signature
    std::vector<std::shared_ptr<complex4DReg>> model; // Renamed 'slow_den' to 'model'

    nx = 500;
    auto ax1 = axis(nx, 0.f, 10.0f); // Example: 10m spacing
    ny = 200;
    auto ax2 = axis(ny, 0.f, 10.0f);
    nw = 100; // Total frequencies
    auto ax3 = axis(nw, 1.f, 1.f); // Freq: 1Hz to 10Hz
    ns = 1; 
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
    root["ginsu_x"] = 0.0; // Padding in meters
    root["ginsu_y"] = 0.0;
    root["look_ahead"] = look_ahead;
    root["compress_error"] = 1e-6;
    root["wflds_to_store"] = 4;
    par = std::make_shared<jsonParamObj>(root); // Assuming constructor exists

    int n_unique_src = ns; // Define number of unique shots/IDs
    int nsrc_traces = n_unique_src; // Assuming one trace per unique source ID for simplicity here
    int nrec_traces = n_unique_src * 100; // e.g., 100 receivers per shot

    // Domain/Range Hypercubes: Assuming [Freq, Trace]
    // Make sure axis indices match constructor/subHyper logic (0=Freq, 1=Trace)
    // Using nw (frequency count) and trace counts
    domainHyper = createHypercube({ax[2], axis(nsrc_traces, 0, 1, "trace")});
    rangeHyper = createHypercube({ax[2], axis(nrec_traces, 0, 1, "receiver")});

    // Create predictable test coords instead of random
    createTestCoords(nsrc_traces, n_unique_src, nx * ax1.d, ny * ax2.d, 5.0f,
                        sx_all, sy_all, sz_all, s_ids_all);
    createTestCoords(nrec_traces, n_unique_src, nx * ax1.d, ny * ax2.d, 10.0f,
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

    std::cout << "Setting up StreamingPropagator..." << std::endl;
    // Use the parsed/default 'batches' vector from command line
    StreamingPropagator streamer(domainHyper, rangeHyper, slowHyper, wavelet,
        sx_all, sy_all, sz_all, s_ids_all,
        rx_all, ry_all, rz_all, r_ids_all,
        par, batches); // Pass the vector here

    std::cout << "Running forward propagation..." << std::endl;

    // --- Profiling Section using NVTX ---
    // This marks the region in Nsight Systems / other NVTX-aware profilers
    nvtxRangePushA("StreamingPropagator::forward"); // Start NVTX range marker (ASCII version)
    
    auto start_cpu_time = std::chrono::steady_clock::now();
    streamer.forward(false, model, data); // Call the method to be profiled
    auto end_cpu_time = std::chrono::steady_clock::now();

    nvtxRangePop(); // End NVTX range marker
    // --- End Profiling Section ---

    std::cout << "Forward propagation finished." << std::endl;

    // --- Calculate and Report CPU Duration ---
    auto duration_cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu_time - start_cpu_time);
    auto duration_cpu_s = std::chrono::duration_cast<std::chrono::duration<double>>(end_cpu_time - start_cpu_time); // For seconds with fraction

    std::cout << "------------------------------------------" << std::endl;
    std::cout << "CPU Time for forward(): " << duration_cpu_ms.count() << " ms"
              << " (" << duration_cpu_s.count() << " s)" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    return 0; // Indicate successful execution
}
