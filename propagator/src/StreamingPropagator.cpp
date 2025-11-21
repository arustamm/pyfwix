#include <Propagator.h>
#include <vector>
#include <memory>
#include <future>
#include <unordered_map>
#include <thread>
#include <StreamingPropagator.h>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>


StreamingPropagator::StreamingPropagator(
    const std::shared_ptr<hypercube>& domain,
    const std::shared_ptr<hypercube>& range,
    std::shared_ptr<hypercube> slow_hyper,
    std::shared_ptr<complex2DReg> wavelet,
    const std::vector<float>& sx, const std::vector<float>& sy, const std::vector<float>& sz,
    const std::vector<int>& s_ids,
    const std::vector<float>& rx, const std::vector<float>& ry, const std::vector<float>& rz,
    const std::vector<int>& r_ids,
    std::shared_ptr<paramObj> par,
    const std::vector<int>& num_batches) 
{
    // Store dimensions
    sx_all = sx;
    sy_all = sy;
    sz_all = sz;
    src_ids_all = s_ids;
    rx_all = rx;
    ry_all = ry;
    rz_all = rz;
    r_ids_all = r_ids;
    // get padding parameters used per batch of sources and receivers
    ginsu_x = par->getFloat("ginsu_x", 0.f);
    ginsu_y = par->getFloat("ginsu_y", 0.f);
    
    // Store batch information
    nsrc_batches = num_batches[0];
    nfreq_batches = num_batches[1];
    int total_freq = domain->getAxis(1).n; 
    if (total_freq < nfreq_batches) {
      std::cerr << "WARNING: Number of frequencies (" << total_freq << ") is less than the number of batches (" << nfreq_batches << "). Reducing number of batches..." << std::endl;
      nfreq_batches = total_freq;
    }
    
    // Extract dimensions
    
    nx = slow_hyper->getAxis(1).n;
    ny = slow_hyper->getAxis(2).n;
    nw = slow_hyper->getAxis(3).n;
    nz = slow_hyper->getAxis(4).n;
    
    // Divide sources into batches
    divideSourcesIntoBatches();
    
    // Divide frequencies into batches
    divideFrequenciesIntoBatches();
    
    // Create CUDA streams - one for each propagator
    createCudaStreams();
    
    // Create propagators for each source batch and frequency batch
    createPropagators(domain, range, slow_hyper, wavelet, par);
}

void StreamingPropagator::divideSourcesIntoBatches() {
  // Count unique sources
  std::unordered_set<int> unique_sources(src_ids_all.begin(), src_ids_all.end());
  int total_sources = unique_sources.size();

  if (total_sources < nsrc_batches) {
    std::cerr << "WARNING: Number of unique sources (" << total_sources << ") is less than the number of batches (" << nsrc_batches << "). Reducing number of batches..." << std::endl;
    nsrc_batches = total_sources;
  }

  // Divide the sources as evenly as possible
  std::vector<int> src_ids_unique(unique_sources.begin(), unique_sources.end());
  std::sort(src_ids_unique.begin(), src_ids_unique.end());
  
  int sources_per_batch = total_sources / nsrc_batches;
  int remainder = total_sources % nsrc_batches;
  
  // Create source batches
  sx_batches.resize(nsrc_batches);
  sy_batches.resize(nsrc_batches);
  sz_batches.resize(nsrc_batches);
  src_ids_batches.resize(nsrc_batches);
  src_index_batches.resize(nsrc_batches);
  
  // Also create receiver batches
  rx_batches.resize(nsrc_batches);
  ry_batches.resize(nsrc_batches);
  rz_batches.resize(nsrc_batches);
  r_ids_batches.resize(nsrc_batches);
  r_index_batches.resize(nsrc_batches);
  
  // Create a mapping from source ID to indices in the source arrays
  for (int i = 0; i < src_ids_all.size(); ++i) {
      src_id_to_indices[src_ids_all[i]].push_back(i);
  }
  
  // Create a mapping from source ID to indices in the receiver arrays
  std::unordered_map<int, std::vector<int>> src_id_to_rec_indices;
  for (int i = 0; i < r_ids_all.size(); ++i) {
      src_id_to_rec_indices[r_ids_all[i]].push_back(i);
  }
  
  // Assign sources to batches
  size_t start_idx = 0;
  for (int batch = 0; batch < nsrc_batches; ++batch) {
    int batch_size = sources_per_batch + (batch < remainder ? 1 : 0);
    
    for (int i = 0; i < batch_size && i < src_ids_unique.size(); ++i) {
      int src_id = src_ids_unique[start_idx];
      src_to_propagator[src_id] = batch;
      
      // Add all source instances with this ID to the batch
      if (src_id_to_indices.find(src_id) != src_id_to_indices.end()) {
        for (int idx : src_id_to_indices[src_id]) {
          sx_batches[batch].push_back(sx_all[idx]);
          sy_batches[batch].push_back(sy_all[idx]);
          sz_batches[batch].push_back(sz_all[idx]);
          src_ids_batches[batch].push_back(src_ids_all[idx]);
          src_index_batches[batch].push_back(idx);
        }
      }
      
      // Add all receiver instances with this ID to the batch
      if (src_id_to_rec_indices.find(src_id) != src_id_to_rec_indices.end()) {
        for (int idx : src_id_to_rec_indices[src_id]) {
          rx_batches[batch].push_back(rx_all[idx]);
          ry_batches[batch].push_back(ry_all[idx]);
          rz_batches[batch].push_back(rz_all[idx]);
          r_ids_batches[batch].push_back(r_ids_all[idx]);
          r_index_batches[batch].push_back(idx);
        }
      }

      start_idx++;
    }
  }
  
  // Check if all batches have both sources and receivers
  for (int batch = 0; batch < nsrc_batches; ++batch) {
      if (src_ids_batches[batch].empty()) {
          std::cout << "Warning: Batch " << batch << " has no sources" << std::endl;
      }
      if (r_ids_batches[batch].empty()) {
          std::cout << "Warning: Batch " << batch << " has no receivers" << std::endl;
      }
  }
}

void StreamingPropagator::divideFrequenciesIntoBatches() {
  // Calculate frequencies per batch
  int freqs_per_batch = nw / nfreq_batches;
  int remainder = nw % nfreq_batches;
  
  freq_start_indices.resize(nfreq_batches);
  freq_batch_sizes.resize(nfreq_batches);
  
  int start_idx = 0;
  for (int batch = 0; batch < nfreq_batches; ++batch) {
    freq_start_indices[batch] = start_idx;
    freq_batch_sizes[batch] = freqs_per_batch + (batch < remainder ? 1 : 0);
    start_idx += freq_batch_sizes[batch];
  }
}

void StreamingPropagator::createCudaStreams() {
  // Create one stream for each propagator
  streams.resize(nsrc_batches * nfreq_batches);
  for (int i = 0; i < nsrc_batches * nfreq_batches; ++i) 
    CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
}

void StreamingPropagator::createPropagators(
    const std::shared_ptr<hypercube>& domain,
    const std::shared_ptr<hypercube>& range,
    std::shared_ptr<hypercube> slow_hyper,
    std::shared_ptr<complex2DReg> wavelet,
    std::shared_ptr<paramObj> par)
{
  propagators.resize(nsrc_batches * nfreq_batches);
  
  for (int src_batch = 0; src_batch < nsrc_batches; ++src_batch) {
    for (int freq_batch = 0; freq_batch < nfreq_batches; ++freq_batch) {
      int prop_idx = src_batch * nfreq_batches + freq_batch;
      
      // Create frequency-sliced domain and range hypercubes
      int start_freq = freq_start_indices[freq_batch];
      this->start_freqs.push_back(start_freq);

      int batch_freq_size = freq_batch_sizes[freq_batch];
      
      auto batch_domain = createSubHypercube(domain, src_index_batches[src_batch], start_freq, batch_freq_size);
      auto batch_range = createSubHypercube(range, r_index_batches[src_batch], start_freq, batch_freq_size);
      auto batch_slow_hyper = createSubSlowness(slow_hyper, start_freq, batch_freq_size, src_batch);
      // Extract frequency-sliced wavelet
      auto batch_wavelet = createSubWavelet(wavelet, batch_domain, src_index_batches[src_batch], start_freq);
      // allocate space for model_batch
      std::vector<std::shared_ptr<complex4DReg>> model;
      model.push_back(std::make_shared<complex4DReg>(batch_slow_hyper));
      model.push_back(std::make_shared<complex4DReg>(batch_slow_hyper));
      model_batches.push_back(model);
      // allocate space for data_batch
      data_batches.push_back(std::make_shared<complex2DReg>(batch_range));
      
      // Create propagator
      propagators[prop_idx] = std::make_unique<Propagator>(
          batch_domain,
          batch_range,
          batch_slow_hyper,
          batch_wavelet,
          sx_batches[src_batch],
          sy_batches[src_batch],
          sz_batches[src_batch],
          src_ids_batches[src_batch],
          rx_batches[src_batch],
          ry_batches[src_batch],
          rz_batches[src_batch],
          r_ids_batches[src_batch],
          par,
          nullptr,
          nullptr,
          dim3(16,16,4),
          dim3(16,16,4),
          streams[prop_idx]
      );
    }
  }
}

std::shared_ptr<hypercube> StreamingPropagator::createSubHypercube(
    const std::shared_ptr<hypercube>& original,
    const std::vector<int>& ids,
    int start_freq,
    int freq_batch_size)
{
  // Create batch-specific domain (source traces for this batch)
  auto axes = original->getAxes();
  // Adjust number of source traces for this batch
  axes[1].n = ids.size();
  // keep the same origin equal to 0
  // Adjust frequency axis
  axes[0].n = freq_batch_size;
  axes[0].o = axes[0].o + start_freq * axes[0].d;
  
  return std::make_shared<hypercube>(axes);
}

std::shared_ptr<hypercube> StreamingPropagator::createSubSlowness(
  const std::shared_ptr<hypercube>& original,
  int start_freq,
  int freq_batch_size,
  int src_batch)
{
  // First calculate the spatial windowing parameters based on acquisition geometry
  int min_ix, max_ix, min_iy, max_iy;
  std::tie(min_ix, max_ix, min_iy, max_iy) = calculateWindowParameters(src_batch, original);
  this->minx.push_back(min_ix);
  this->miny.push_back(min_iy);
  
  // Calculate window sizes
  int window_nx = max_ix - min_ix + 1;
  int window_ny = max_iy - min_iy + 1;
  
  // Create batch-specific slowness hypercube
  auto axes = original->getAxes();
  
  // Adjust x and y dimensions based on windowing
  axes[0].n = window_nx;
  axes[0].o = original->getAxis(1).o + min_ix * original->getAxis(1).d;
  
  axes[1].n = window_ny;
  axes[1].o = original->getAxis(2).o + min_iy * original->getAxis(2).d;
  
  // Ad2ust frequency axis
  axes[2].n = freq_batch_size;
  axes[2].o = original->getAxis(3).o + start_freq * original->getAxis(3).d;
  
  // Create the new hypercube with windowed dimensions
  auto result = std::make_shared<hypercube>(axes);
  
  // std::cout << "Created windowed slowness hypercube:" << std::endl
  //           << "  Original dimensions: x=" << original->getAxis(1).n 
  //           << ", y=" << original->getAxis(2).n
  //           << ", w=" << original->getAxis(3).n
  //           << ", z=" << original->getAxis(4).n << std::endl
  //           << "  Windowed dimensions: x=" << axes[1].n 
  //           << ", y=" << axes[2].n
  //           << ", w=" << axes[3].n
  //           << ", z=" << axes[4].n << std::endl;
  
  return result;
}

std::shared_ptr<complex2DReg> StreamingPropagator::createSubWavelet(
  const std::shared_ptr<complex2DReg>& wavelet,
  const std::shared_ptr<hypercube>& subdomain, 
  const std::vector<int>& trace_indices,
  const int& start_freq
)
{
  auto sub_wavelet = std::make_shared<complex2DReg>(subdomain);

  int nfreq = subdomain->getAxis(1).n;
  int ntraces = wavelet->getHyper()->getAxis(2).n;
  
  // Extract the subset of the wavelet data
  for (int i = 0; i < trace_indices.size(); ++i) {
    int idx = trace_indices[i];
    if (idx < 0 || idx >= ntraces) 
        std::runtime_error("Creating subwavelet: Trace index out of bounds");
    // Copy the frequency slice for this source
    for (int iw = 0; iw < nfreq; ++iw) {
        (*sub_wavelet->_mat)[i][iw] = (*wavelet->_mat)[idx][start_freq + iw];
    }
  }
  
  return sub_wavelet;
}

// Calculate windowing parameters for a specific source batch
std::tuple<int, int, int, int> StreamingPropagator::calculateWindowParameters(
  int src_batch, 
  const std::shared_ptr<hypercube>& model_hyper) 
{
  // Find min/max coordinates for sources in this batch
  float min_sx = std::numeric_limits<float>::max();
  float max_sx = std::numeric_limits<float>::lowest();
  float min_sy = std::numeric_limits<float>::max();
  float max_sy = std::numeric_limits<float>::lowest();
  
  for (int i = 0; i < sx_batches[src_batch].size(); ++i) {
    min_sx = std::min(min_sx, sx_batches[src_batch][i]);
    max_sx = std::max(max_sx, sx_batches[src_batch][i]);
    min_sy = std::min(min_sy, sy_batches[src_batch][i]);
    max_sy = std::max(max_sy, sy_batches[src_batch][i]);
  }
  
  // Find min/max coordinates for receivers in this batch
  float min_rx = std::numeric_limits<float>::max();
  float max_rx = std::numeric_limits<float>::lowest();
  float min_ry = std::numeric_limits<float>::max();
  float max_ry = std::numeric_limits<float>::lowest();
  
  for (int i = 0; i < rx_batches[src_batch].size(); ++i) {
    min_rx = std::min(min_rx, rx_batches[src_batch][i]);
    max_rx = std::max(max_rx, rx_batches[src_batch][i]);
    min_ry = std::min(min_ry, ry_batches[src_batch][i]);
    max_ry = std::max(max_ry, ry_batches[src_batch][i]);
  }
  
  // Calculate overall min/max coordinates 
  float min_x = std::min(min_sx, min_rx);
  float max_x = std::max(max_sx, max_rx);
  float min_y = std::min(min_sy, min_ry);
  float max_y = std::max(max_sy, max_ry);
  
  // Add absolute padding in respective units
  min_x -= this->ginsu_x;
  max_x += this->ginsu_x;
  min_y -= this->ginsu_y;
  max_y += this->ginsu_y;
  
  // Convert coordinates to model indices
  // Assuming model coordinates increase with index
  int model_nx = model_hyper->getAxis(1).n;
  int model_ny = model_hyper->getAxis(2).n;
  float x_orig = model_hyper->getAxis(1).o;
  float y_orig = model_hyper->getAxis(2).o;
  float dx = model_hyper->getAxis(1).d;
  float dy = model_hyper->getAxis(2).d;
  
  // Calculate the indices with proper bounds checking
  int min_ix = std::max(0, static_cast<int>(floor((min_x - x_orig) / dx)));
  int max_ix = std::min(model_nx - 1, static_cast<int>(ceil((max_x - x_orig) / dx)));
  int min_iy = std::max(0, static_cast<int>(floor((min_y - y_orig) / dy)));
  int max_iy = std::min(model_ny - 1, static_cast<int>(ceil((max_y - y_orig) / dy)));
  
  // Ensure we have at least some window (in case no sources or receivers are in this batch)
  if (max_ix <= min_ix) {
      min_ix = 0;
      max_ix = model_nx - 1;
  }
  if (max_iy <= min_iy) {
      min_iy = 0;
      max_iy = model_ny - 1;
  }
  
  // Log the coordinate and index ranges
  // std::cout << "Window for batch " << src_batch << ":" << std::endl
  //           << "  Coordinate range: x[" << min_x << " - " << max_x << "], y[" << min_y << " - " << max_y << "]" << std::endl
  //           << "  Index range: x[" << min_ix << " - " << max_ix << "], y[" << min_iy << " - " << max_iy << "]" << std::endl
  //           << "  Window size: " << (max_ix - min_ix + 1) << " x " << (max_iy - min_iy + 1) << std::endl;
  
  return std::make_tuple(min_ix, max_ix, min_iy, max_iy);
}

void StreamingPropagator::windowModel(
  const std::vector<std::shared_ptr<complex4DReg>>& original,
  std::vector<std::shared_ptr<complex4DReg>>& model_batch,
  int min_ix, int min_iy, 
  int start_freq
) {

  auto ax = model_batch[0]->getHyper()->getAxes();

  // Copy the data from the original model to the windowed model
  // Assuming the order is [iz][iw][iy][ix] in the 4D array
  for (int m = 0; m < original.size(); ++m) {
    tbb::parallel_for(
      tbb::blocked_range2d<int, int>(0, ax[3].n, 0, ax[2].n),
      [&](const tbb::blocked_range2d<int>& r) {
        for (int i3 = r.rows().begin(); i3 < r.rows().end(); ++i3) {
          for (int i2 = r.cols().begin(); i2 < r.cols().end(); ++i2) {
            for (int i1 = 0; i1 < ax[1].n; ++i1) {
              for (int i0 = 0; i0 < ax[0].n; ++i0) {
                // Access source and destination using multi_array indexing
                (*model_batch[m]->_mat)[i3][i2][i1][i0] = 
                    (*original[m]->_mat)[i3][start_freq + i2][min_iy + i1][min_ix + i0];
              }
            }
          }
        }
      });
  }

}

void StreamingPropagator::forward(bool add, std::vector<std::shared_ptr<complex4DReg>> model, std::shared_ptr<complex2DReg> data) {
  // Clear the output data if not adding
  if (!add) data->zero();
  
  tbb::parallel_for(
    tbb::blocked_range<int>(0, nsrc_batches * nfreq_batches),
    [&](const tbb::blocked_range<int>& r) {
      for (int i = r.begin(); i != r.end(); ++i) {
        // Create windowed models
        windowModel(model, model_batches[i], minx[i], miny[i], start_freqs[i]);
        propagators[i]->forward(true, model_batches[i], data_batches[i]);

        int src_batch = i / nfreq_batches;
        int freq_batch = i % nfreq_batches;
        
        int start_freq = freq_start_indices[freq_batch];
        int batch_freq_size = freq_batch_sizes[freq_batch];

        CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
        
        for (int j = 0; j < r_index_batches[src_batch].size(); j++) {
          int idx = r_index_batches[src_batch][j];
          for (int iw = 0; iw < batch_freq_size; iw++) {
            (*data->_mat)[idx][start_freq + iw] += 
              (*data_batches[i]->_mat)[j][iw];
          }
        }

      }
    }, tbb::static_partitioner()
  );
}
