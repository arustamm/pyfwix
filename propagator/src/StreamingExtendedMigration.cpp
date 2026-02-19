#include "StreamingExtendedMigration.h"

StreamingExtendedMigration::StreamingExtendedMigration(
    std::shared_ptr<StreamingPropagator> streaming_prop,
    const std::vector<std::shared_ptr<complex4DReg>>& slow_den
) : _streaming_prop(streaming_prop) 
{
    createMigrationOperators(slow_den);
}

void StreamingExtendedMigration::createMigrationOperators(const std::vector<std::shared_ptr<complex4DReg>>& slow_den) {
    int n_ops = _streaming_prop->propagators.size();
    migration_operators.resize(n_ops);
    image_batches.resize(n_ops);
    slow_batches.resize(n_ops);
    data_batches.resize(n_ops);

    for (int i = 0; i < n_ops; ++i) {
        auto prop = _streaming_prop->propagators[i];
        
        int min_x = _streaming_prop->minx[i]; 
        int min_y = _streaming_prop->miny[i];
        int start_f = _streaming_prop->start_freqs[i];
        int freq_size = _streaming_prop->freq_batch_sizes[i % _streaming_prop->nfreq_batches];

        // Create the Ginsu-windowed Hypercube for this batch
        auto batch_hyper = _streaming_prop->createSubSlowness(
            slow_den[0]->getHyper(), start_f, freq_size, i / _streaming_prop->nfreq_batches);

        // Allocate local image and data workspace
        image_batches[i] = std::make_shared<complex4DReg>(batch_hyper);
        data_batches[i] = std::make_shared<complex2DReg>(prop->getRange());

        // Prepare and window background models
        slow_batches[i].push_back(std::make_shared<complex4DReg>(batch_hyper));
        slow_batches[i].push_back(std::make_shared<complex4DReg>(batch_hyper));
        _streaming_prop->windowModel(slow_den, slow_batches[i], min_x, min_y, start_f);

        // Initialize the single-batch Migration operator
        migration_operators[i] = std::make_unique<ExtendedMigration>(
            batch_hyper,
            prop->getRange(),
            slow_batches[i],
            prop,
            dim3(16, 16, 4),
            dim3(16, 16, 4),
            _streaming_prop->streams[i]
        );
    }
}


void StreamingExtendedMigration::migrate(bool add, std::shared_ptr<complex4DReg> image, std::shared_ptr<complex2DReg> data) {
    if (!add) image->zero();

    tbb::parallel_for(
        tbb::blocked_range<int>(0, migration_operators.size()),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                
                // Get batch metadata from the streaming propagator
                int src_batch = i / _streaming_prop->nfreq_batches;
                int freq_batch = i % _streaming_prop->nfreq_batches;
                int min_x = _streaming_prop->minx[i];
                int min_y = _streaming_prop->miny[i];
                int start_f = _streaming_prop->start_freqs[i];
                int batch_freq_size = _streaming_prop->freq_batch_sizes[freq_batch];

                // 1. Prepare Local Data (Residuals)
                data_batches[i]->zero(); 
                auto& r_indices = _streaming_prop->r_index_batches[src_batch];
                
                for (int j = 0; j < r_indices.size(); j++) {
                    int idx = r_indices[j];
                    for (int iw = 0; iw < batch_freq_size; iw++) {
                        (*data_batches[i]->_mat)[j][iw] = (*data->_mat)[idx][start_f + iw];
                    }
                }

                // 2. Run Migration (The Adjoint call)
                // Note: image_batches[i] is already windowed to the local Ginsu size
                migration_operators[i]->migrate(false, image_batches[i], data_batches[i]);

                // 3. Sync Stream
                CHECK_CUDA_ERROR(cudaStreamSynchronize(_streaming_prop->streams[i]));

                // 4. Accumulate Image
                // We lock because multiple shot/freq batches might overlap in the global image
                std::lock_guard<std::mutex> lock(accum_mutex);
                accumulateImage(image, image_batches[i], min_x, min_y, start_f);
            }
        }, tbb::static_partitioner()
    );
}

void StreamingExtendedMigration::accumulateImage(
    std::shared_ptr<complex4DReg>& global_image,
    const std::shared_ptr<complex4DReg>& batch_image,
    int min_ix, int min_iy, int start_freq) 
{
    auto ax = batch_image->getHyper()->getAxes();

    // Parallelize accumulation over depth (i3) and frequency (i2)
    tbb::parallel_for(
        tbb::blocked_range2d<int, int>(0, ax[3].n, 0, ax[2].n),
        [&](const tbb::blocked_range2d<int>& r) {
            for (int i3 = r.rows().begin(); i3 < r.rows().end(); ++i3) {
                for (int i2 = r.cols().begin(); i2 < r.cols().end(); ++i2) {
                    for (int i1 = 0; i1 < ax[1].n; ++i1) {
                        for (int i0 = 0; i0 < ax[0].n; ++i0) {
                            (*global_image->_mat)[i3][start_freq + i2][min_iy + i1][min_ix + i0] += 
                                (*batch_image->_mat)[i3][i2][i1][i0];
                        }
                    }
                }
            }
        }
    );
}