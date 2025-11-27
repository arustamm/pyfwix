#include "StreamingExtendedBorn.h"

StreamingExtendedBorn::StreamingExtendedBorn(
    std::shared_ptr<StreamingPropagator> streaming_prop,
    const std::vector<std::shared_ptr<complex4DReg>>& slow_den
) : _streaming_prop(streaming_prop) 
{
    createBornOperators(slow_den);
}

void StreamingExtendedBorn::createBornOperators(const std::vector<std::shared_ptr<complex4DReg>>& slow_den) {
    int n_ops = _streaming_prop->propagators.size();
    born_operators.resize(n_ops);
    
    // Resize batch storage
    model_batches.resize(n_ops);
    slow_batches.resize(n_ops);
    data_batches.resize(n_ops);

    // We iterate through the existing propagators in the StreamingPropagator
    // and wrap each one in an ExtendedBorn operator.
    for (int i = 0; i < n_ops; ++i) {
        auto prop = _streaming_prop->propagators[i];
        
        // 1. Get batch geometry info from the streaming propagator
        int min_x = _streaming_prop->minx[i]; 
        int min_y = _streaming_prop->miny[i];
        int start_f = _streaming_prop->start_freqs[i];
        int freq_size = _streaming_prop->freq_batch_sizes[i % _streaming_prop->nfreq_batches];

        // 2. Create windowed versions of the background slowness/density
        // We can reuse the windowing logic logic from StreamingPropagator if available, 
        // or implement a local helper.
        auto batch_slow = _streaming_prop->createSubSlowness(slow_den[0]->getHyper(), start_f, freq_size, i / _streaming_prop->nfreq_batches);

        // Allocate batch model vector
        model_batches[i].push_back(std::make_shared<complex4DReg>(batch_slow));
        model_batches[i].push_back(std::make_shared<complex4DReg>(batch_slow));
        slow_batches[i].push_back(std::make_shared<complex4DReg>(batch_slow));
        slow_batches[i].push_back(std::make_shared<complex4DReg>(batch_slow));
        _streaming_prop->windowModel(slow_den, slow_batches[i], min_x, min_y, start_f);

        // 3. Create batch data vector (matching the range of the specific propagator)
        data_batches[i] = std::make_shared<complex2DReg>(prop->getRange());

        // 4. Create the ExtendedBorn operator
        // We pass the specific stream used by this propagator to ensure synchronization
        born_operators[i] = std::make_unique<ExtendedBorn>(
            batch_slow,
            prop->getRange(),
            slow_batches[i], // Initial background model
            prop,            // The single-batch propagator
            dim3(16, 16, 4),
            dim3(16, 16, 4),
            _streaming_prop->streams[i] // Reuse the stream!
        );
    }
}

// void StreamingExtendedBorn::set_background_model(std::vector<std::shared_ptr<complex4DReg>> model) {
//     // This updates the background velocity in the underlying propagators
//     _streaming_prop->forward(false, model, nullptr); // This calls set_background internally or you can expose it
    
//     // Also need to update the local batch background models for ExtendedBorn logic
//     tbb::parallel_for(
//         tbb::blocked_range<int>(0, born_operators.size()),
//         [&](const tbb::blocked_range<int>& r) {
//             for (int i = r.begin(); i != r.end(); ++i) {
//                 int min_x = _streaming_prop->minx[i];
//                 int min_y = _streaming_prop->miny[i];
//                 int start_f = _streaming_prop->start_freqs[i];
                
//                 windowModel(model, model_batches[i], min_x, min_y, start_f);
//                 born_operators[i]->set_background_model(model_batches[i]);
//             }
//         }
//     );
// }

void StreamingExtendedBorn::forward(bool add, std::vector<std::shared_ptr<complex4DReg>> model, std::shared_ptr<complex2DReg> data) {
    if (!add) data->zero();

    tbb::parallel_for(
        tbb::blocked_range<int>(0, born_operators.size()),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                
                // 1. Window the input perturbation model
                int min_x = _streaming_prop->minx[i];
                int min_y = _streaming_prop->miny[i];
                int start_f = _streaming_prop->start_freqs[i];
                _streaming_prop->windowModel(model, model_batches[i], min_x, min_y, start_f);

                // 2. Run Forward Extended Born
                born_operators[i]->forward(false, model_batches[i], data_batches[i]);

                // 3. Sync Stream before accumulation
                CHECK_CUDA_ERROR(cudaStreamSynchronize(_streaming_prop->streams[i]));

                // 4. Accumulate into global data
                
                int src_batch = i / _streaming_prop->nfreq_batches;
                int freq_batch = i % _streaming_prop->nfreq_batches;
                int batch_freq_size = _streaming_prop->freq_batch_sizes[freq_batch];
                auto& r_indices = _streaming_prop->r_index_batches[src_batch];

                for (int j = 0; j < r_indices.size(); j++) {
                    int idx = r_indices[j];
                    for (int iw = 0; iw < batch_freq_size; iw++) {
                        (*data->_mat)[idx][start_f + iw] += (*data_batches[i]->_mat)[j][iw];
                    }
                }
            }
        }, tbb::static_partitioner()
    );
}

void StreamingExtendedBorn::adjoint(bool add, std::vector<std::shared_ptr<complex4DReg>> model, std::shared_ptr<complex2DReg> data) {
    if (!add) 
        for(auto& m : model) m->zero();

    tbb::parallel_for(
        tbb::blocked_range<int>(0, born_operators.size()),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                
                int src_batch = i / _streaming_prop->nfreq_batches;
                int freq_batch = i % _streaming_prop->nfreq_batches;
                int min_x = _streaming_prop->minx[i];
                int min_y = _streaming_prop->miny[i];
                int start_f = _streaming_prop->start_freqs[i];
                int batch_freq_size = _streaming_prop->freq_batch_sizes[freq_batch];

                // 1. Window the input data (residuals)
                data_batches[i]->zero(); // Ensure clean slate
                auto& r_indices = _streaming_prop->r_index_batches[src_batch];
                
                // Copy global data to local batch data
                for (int j = 0; j < r_indices.size(); j++) {
                    int idx = r_indices[j];
                    for (int iw = 0; iw < batch_freq_size; iw++) {
                        (*data_batches[i]->_mat)[j][iw] = (*data->_mat)[idx][start_f + iw];
                    }
                }

                // 2. Run Adjoint Extended Born
                // The output goes into model_batches[i]
                born_operators[i]->adjoint(false, model_batches[i], data_batches[i]);

                // 3. Sync Stream
                CHECK_CUDA_ERROR(cudaStreamSynchronize(_streaming_prop->streams[i]));

                // 4. Accumulate Gradient (Image)
                // CRITICAL: Imaging condition overlaps in space. Must Lock.
                std::lock_guard<std::mutex> lock(accum_mutex);
                accumulateModel(model, model_batches[i], min_x, min_y, start_f);
            }
        }, tbb::static_partitioner()
    );
}

void StreamingExtendedBorn::accumulateModel(
    std::vector<std::shared_ptr<complex4DReg>>& global_model,
    const std::vector<std::shared_ptr<complex4DReg>>& batch_model,
    int min_ix, int min_iy, int start_freq) 
{
    auto ax = batch_model[0]->getHyper()->getAxes();

    for (int m = 0; m < global_model.size(); ++m) {
      tbb::parallel_for(
      tbb::blocked_range2d<int, int>(0, ax[3].n, 0, ax[2].n),
      [&](const tbb::blocked_range2d<int>& r) {
        for (int i3 = r.rows().begin(); i3 < r.rows().end(); ++i3) {
          for (int i2 = r.cols().begin(); i2 < r.cols().end(); ++i2) {
            for (int i1 = 0; i1 < ax[1].n; ++i1) {
              for (int i0 = 0; i0 < ax[0].n; ++i0) {
                (*global_model[m]->_mat)[i3][start_freq + i2][min_iy + i1][min_ix + i0] += 
                    (*batch_model[m]->_mat)[i3][i2][i1][i0];
              }
            }
          }
        }
      });
    }
}