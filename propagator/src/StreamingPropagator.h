#pragma once
#include <complex4DReg.h>
#include <paramObj.h>
#include "Propagator.h"
#include <future>
#include <unordered_map>
#include <unordered_set>

class StreamingPropagator {
public:
    StreamingPropagator(
        const std::shared_ptr<hypercube>& domain,
        const std::shared_ptr<hypercube>& range,
        std::shared_ptr<hypercube> slow_hyper,
        std::shared_ptr<complex2DReg> wavelet,
        const std::vector<float>& sx, const std::vector<float>& sy, const std::vector<float>& sz,
        const std::vector<int>& s_ids,
        const std::vector<float>& rx, const std::vector<float>& ry, const std::vector<float>& rz,
        const std::vector<int>& r_ids,
        std::shared_ptr<paramObj> par,
        const std::vector<int>& num_batches = {1, 1});

    ~StreamingPropagator() {
        for (auto& stream : streams) {
            CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
        }
    }

    void forward(bool add, std::vector<std::shared_ptr<complex4DReg>> model, std::shared_ptr<complex2DReg> data);

    void divideSourcesIntoBatches();
    void divideFrequenciesIntoBatches();
    void createCudaStreams();
    void createPropagators(
        const std::shared_ptr<hypercube>& domain,
        const std::shared_ptr<hypercube>& range,
        std::shared_ptr<hypercube> slow_hyper,
        std::shared_ptr<complex2DReg> wavelet,
        std::shared_ptr<paramObj> par);

    std::shared_ptr<hypercube> createSubHypercube(
        const std::shared_ptr<hypercube>& original,
        const std::vector<int>& ids,
        int start_freq,
        int freq_batch_size);

    std::shared_ptr<hypercube> createSubSlowness(
        const std::shared_ptr<hypercube>& original,
        int start_freq,
        int freq_batch_size,
        int src_batch
    );

    std::shared_ptr<complex2DReg> createSubWavelet(
        const std::shared_ptr<complex2DReg>& wavelet,
        const std::shared_ptr<hypercube>& subdomain,
        const std::vector<int>& trace_indices,
        const int& start_freq);

    std::tuple<int, int, int, int> calculateWindowParameters(
        int src_batch, 
        const std::shared_ptr<hypercube>& model);

    void windowModel(
        const std::vector<std::shared_ptr<complex4DReg>>& original,
        std::vector<std::shared_ptr<complex4DReg>>& model_batch,
        int min_ix, int min_iy, 
        int start_freq
    );

    // Member variables
    // Store individual propagators
    std::vector<std::unique_ptr<Propagator>> propagators;

    // Data dimensions
    int ntraces, nw, nz, ny, nx;
    
    // Number of frequency batches
    int nfreq_batches;
    
    // Number of source batches
    int nsrc_batches;
    
    // Source information
    std::vector<float> sx_all, sy_all, sz_all;
    std::vector<int> src_ids_all;
    
    // Receiver information
    std::vector<float> rx_all, ry_all, rz_all;
    std::vector<int> r_ids_all;
    
    // For dividing sources into batches
    std::vector<std::vector<float>> sx_batches, sy_batches, sz_batches;
    std::vector<std::vector<int>> src_ids_batches;
    std::vector<std::vector<int>> src_index_batches;
    
    // For dividing receivers into batches
    std::vector<std::vector<float>> rx_batches, ry_batches, rz_batches;
    std::vector<std::vector<int>> r_ids_batches;
    std::vector<std::vector<int>> r_index_batches;
    
    // Store CUDA streams
    std::vector<cudaStream_t> streams;
    
    // Map to track which sources belong to which propagator
    std::unordered_map<int, int> src_to_propagator;
    // Map to track which traces belong to which source id
    std::unordered_map<int, std::vector<int>> src_id_to_indices;
    
    // Frequency batch information
    std::vector<int> freq_start_indices;
    std::vector<int> freq_batch_sizes;

    float ginsu_x, ginsu_y;

    // Batch models and outputs
    std::vector<std::vector<std::shared_ptr<complex4DReg>>> model_batches;
    std::vector<std::shared_ptr<complex2DReg>> data_batches;
    std::vector<int> minx, miny;
    std::vector<int> start_freqs;
};