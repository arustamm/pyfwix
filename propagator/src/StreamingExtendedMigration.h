#pragma once
#include <StreamingPropagator.h>
#include <ExtendedMigration.h>
#include <tbb/tbb.h>
#include <mutex>

namespace SEP {

class StreamingExtendedMigration {
public:
    StreamingExtendedMigration(
        std::shared_ptr<StreamingPropagator> streaming_prop,
        const std::vector<std::shared_ptr<complex4DReg>>& slow_den
    );

    void migrate(bool add, std::shared_ptr<complex4DReg> image, std::shared_ptr<complex2DReg> data);

private:
    std::shared_ptr<StreamingPropagator> _streaming_prop;
    std::vector<std::unique_ptr<ExtendedMigration>> migration_operators;
    
    // Batch storage
    std::vector<std::shared_ptr<complex4DReg>> image_batches; // Local images per stream
    std::vector<std::shared_ptr<complex2DReg>> data_batches;  // Local data per stream
    std::vector<std::vector<std::shared_ptr<complex4DReg>>> slow_batches; // Background models

    std::mutex accum_mutex;
    void createMigrationOperators(const std::vector<std::shared_ptr<complex4DReg>>& slow_den);
    void accumulateImage(std::shared_ptr<complex4DReg>& global_image, 
                         const std::shared_ptr<complex4DReg>& batch_image, 
                         int min_ix, int min_iy, int start_freq);
};

}