#include <WavefieldPool.h>

void WavefieldPool::initialize(
    std::shared_ptr<hypercube> wfld_hyper, 
    std::shared_ptr<paramObj> par,
    std::string run_id,
    int max_depth
) { 
    // 1. Setup Configuration
    _rel_error = par->getFloat("compress_error", 0.0);
    
    // 2. Dimension Folding (Flatten 4D -> 3D)
    auto ax = wfld_hyper->getAxes();
    int nx = ax[0].n;
    int ny = ax[1].n;
    
    // Total floats in the slice (Complex = 2 floats)
    _nbEle = (size_t)wfld_hyper->getN123() * 2; 

    _dims.x = nx * 2; 
    _dims.y = ny;
    _dims.z = _nbEle / (_dims.x * _dims.y); 

    // 3. Allocate Memory
    _ram_storage.resize(max_depth);
    _error_bounds.resize(max_depth);
    
    // Worst-case size is just the raw size
    // This fits both the uncompressed data AND the compressed data (worst case)
    _max_comp_bytes = _nbEle * sizeof(float); 

    // GPU Buffer
    CHECK_CUDA_ERROR(cudaMalloc((void**)&_d_compressed_buffer, _max_comp_bytes));

    // Host Buffer (Pinned for async transfer)
    CHECK_CUDA_ERROR(cudaHostAlloc((void**)&_h_compressed_buffer, _max_comp_bytes, cudaHostAllocDefault));

    // Size Variable (Pinned)
    CHECK_CUDA_ERROR(cudaHostAlloc((void**)&_cmpSize, sizeof(size_t), cudaHostAllocMapped));
}

void WavefieldPool::save_slice(int iz, complex_vector* wfld, cudaStream_t stream) {

    // -------------------------------------------------------------
    // BRANCH 1: NO COMPRESSION (Exact Storage for Dot Test)
    // -------------------------------------------------------------
    if (_rel_error < 1e-16f) {
        size_t raw_size = _nbEle * sizeof(float);
        
        // Copy directly GPU -> Host Pinned Buffer
        // Note: we can skip _d_compressed_buffer and copy straight from wfld->mat
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
            _h_compressed_buffer, 
            wfld->mat, 
            raw_size, 
            cudaMemcpyDeviceToHost, 
            stream
        ));

        // Wait for transfer
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        // Store in RAM
        _ram_storage[iz].resize(raw_size);
        std::memcpy(_ram_storage[iz].data(), _h_compressed_buffer, raw_size);
        
        // We don't need to store an error bound here
        return; 
    }

    // -------------------------------------------------------------
    // BRANCH 2: COMPRESSION (cuSZp)
    // -------------------------------------------------------------

    auto [min_val, max_val] = wfld->getMinMax();
    float range = max_val - min_val;
    if (range < 1e-20f) range = 1.0f;

    float abs_error_bound = range * _rel_error;
    _error_bounds[iz] = abs_error_bound;
    
    cuSZp_compress_3D_outlier_f32(
        (float*)wfld->mat,        
        _d_compressed_buffer,      
        _nbEle,                    
        _cmpSize,                  
        _dims,                     
        abs_error_bound,              
        stream
    );

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    size_t actual_size = *_cmpSize;

    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        _h_compressed_buffer, 
        _d_compressed_buffer, 
        actual_size, 
        cudaMemcpyDeviceToHost, 
        stream
    ));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    
    _ram_storage[iz].resize(actual_size);
    std::memcpy(_ram_storage[iz].data(), _h_compressed_buffer, actual_size);
}

void WavefieldPool::load_slice(int iz, complex_vector* wfld, cudaStream_t stream) {
    
    size_t data_size = _ram_storage[iz].size();
    unsigned char* src_ptr = _ram_storage[iz].data();

    if (data_size == 0) {
        throw std::runtime_error("Attempted to load empty slice at iz=" + std::to_string(iz));
    }

    // Copy RAM -> Pinned Buffer (CPU copy)
    std::memcpy(_h_compressed_buffer, src_ptr, data_size);

    // -------------------------------------------------------------
    // BRANCH 1: NO COMPRESSION (Exact Storage)
    // -------------------------------------------------------------
    if (_rel_error < 1e-16f) {
        // Copy directly Host Pinned -> GPU Destination
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
            wfld->mat, 
            _h_compressed_buffer, 
            data_size, 
            cudaMemcpyHostToDevice, 
            stream
        ));
        
        // Sync to protect the host buffer from being overwritten
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        return;
    }

    // -------------------------------------------------------------
    // BRANCH 2: DECOMPRESSION (cuSZp)
    // -------------------------------------------------------------

    // Copy Pinned -> GPU Temp Buffer
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        _d_compressed_buffer,
        _h_compressed_buffer, 
        data_size, 
        cudaMemcpyHostToDevice, 
        stream
    ));

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    cuSZp_decompress_3D_outlier_f32(
        (float*)wfld->mat,         
        _d_compressed_buffer,       
        _nbEle,                     
        data_size,            
        _dims,                      
        _error_bounds[iz],               
        stream
    );
}

void WavefieldPool::cleanup() {
    if (_d_compressed_buffer) cudaFree(_d_compressed_buffer);
    if (_h_compressed_buffer) cudaFreeHost(_h_compressed_buffer);
    if (_cmpSize) cudaFreeHost(_cmpSize);
    _ram_storage.clear();
    _error_bounds.clear();
}