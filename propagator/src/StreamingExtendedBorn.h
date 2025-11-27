#pragma once
#include <ExtendedBorn.h>
#include <StreamingPropagator.h>
#include <mutex>
#include <tbb/tbb.h>

class StreamingExtendedBorn {
public:
    // Constructor
    StreamingExtendedBorn(
        std::shared_ptr<StreamingPropagator> streaming_prop,
        const std::vector<std::shared_ptr<complex4DReg>>& slow_den
    );

    // Forward: Perturbation Model -> Data Residuals
    void forward(bool add, std::vector<std::shared_ptr<complex4DReg>> model, std::shared_ptr<complex2DReg> data);

    // Adjoint: Data Residuals -> Perturbation Model (Image)
    void adjoint(bool add, std::vector<std::shared_ptr<complex4DReg>> model, std::shared_ptr<complex2DReg> data);

    // Update the background model (slowness/density) in the underlying propagators
    // void set_background_model(std::vector<std::shared_ptr<complex4DReg>> model);

private:
    // The underlying streaming propagator (holds geometry, streams, and background wavefields)
    std::shared_ptr<StreamingPropagator> _streaming_prop;

    // Collection of single-batch ExtendedBorn operators
    std::vector<std::unique_ptr<ExtendedBorn>> born_operators;

    // Storage for batched inputs/outputs
    // [batch_id][0=slow, 1=den]
    std::vector<std::vector<std::shared_ptr<complex4DReg>>> slow_batches; 
    std::vector<std::vector<std::shared_ptr<complex4DReg>>> model_batches; 
    std::vector<std::shared_ptr<complex2DReg>> data_batches;

    // Mutex for thread-safe accumulation (Imaging condition)
    std::mutex accum_mutex;

    // Helper to initialize the operators
    void createBornOperators(const std::vector<std::shared_ptr<complex4DReg>>& slow_den);

    // Helper to accumulate windowed gradients back into the global model
    void accumulateModel(
        std::vector<std::shared_ptr<complex4DReg>>& global_model,
        const std::vector<std::shared_ptr<complex4DReg>>& batch_model,
        int min_ix, int min_iy, 
        int start_freq
    );
};