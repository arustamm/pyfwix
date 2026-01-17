#include "WavefieldPool.h"

void WavefieldPool::initialize(
    std::shared_ptr<hypercube> wfld_hyper, 
    std::shared_ptr<paramObj> par,
    std::string run_id,
    int max_depth
) { 
    // 1. Setup Config
    // rate = bits per value. 8.0 is standard for Seismic (4x compression).
    // You can try 12.0 or 16.0 for higher precision.
    double rate = par->getFloat("compress_rate", 8.0);
    
    // 2. Dimensions
    auto ax = wfld_hyper->getAxes();
    int nx = ax[0].n;
    int ny = ax[1].n;
    _nbEle = (size_t)wfld_hyper->getN123() * 2; 

    // ZFP handles 3D arrays natively (nx*2 because complex = 2 floats)
    // Note: ZFP dimensions are (x, y, z)
    int dim_x = nx * 2;
    int dim_y = ny;
    int dim_z = _nbEle / (dim_x * dim_y);

    // 3. Setup ZFP Structures
    _zfield = zfp_field_3d(NULL, zfp_type_float, dim_x, dim_y, dim_z);
    
    _zstream = zfp_stream_open(NULL);
    zfp_stream_set_rate(_zstream, rate, zfp_type_float, 3, 0); 
    
    // CRITICAL: Set execution to CUDA
    if(zfp_stream_set_execution(_zstream, zfp_exec_cuda) == 0) {
        throw std::runtime_error("Failed to set ZFP execution to CUDA! Is ZFP built with -DZFP_WITH_CUDA=ON?");
    }

    // 4. Calculate EXACT Buffer Size
    // ZFP Fixed Rate has a deterministic size. We ask ZFP "how big will it be?"
    size_t bufsize = zfp_stream_maximum_size(_zstream, _zfield);
    _compressed_size_bytes = bufsize;

    // 5. Allocate Buffers (Exact size, no overhead needed)
    CHECK_CUDA_ERROR(cudaMalloc((void**)&_d_compressed_buffer, bufsize));
    CHECK_CUDA_ERROR(cudaHostAlloc((void**)&_h_compressed_buffer, bufsize, cudaHostAllocDefault));
    
    // RAM Storage
    _ram_storage.resize(max_depth);
}

void WavefieldPool::save_slice(int iz, complex_vector* wfld, cudaStream_t stream) {
    
    // 1. Point ZFP Field to the GPU Data
    // We update the pointer (data) but keep dimensions same
    zfp_field_set_pointer(_zfield, (void*)wfld->mat);

    // 2. Point Bitstream to GPU Output Buffer
    // We must re-open the stream for every slice to reset the pointer
    if (_bstream) stream_close(_bstream);
    _bstream = stream_open(_d_compressed_buffer, _compressed_size_bytes);
    zfp_stream_set_bit_stream(_zstream, _bstream);

    size_t size = zfp_compress(_zstream, _zfield);
    
    if (size == 0) {
         throw std::runtime_error("ZFP Compression failed!");
    }

    // 4. Copy to CPU
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        _h_compressed_buffer, 
        _d_compressed_buffer, 
        size, 
        cudaMemcpyDeviceToHost, 
        stream
    ));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 5. Store
    _ram_storage[iz].resize(size);
    std::memcpy(_ram_storage[iz].data(), _h_compressed_buffer, size);
}

void WavefieldPool::load_slice(int iz, complex_vector* wfld, cudaStream_t stream) {
    
    size_t size = _ram_storage[iz].size();
    if (size == 0) throw std::runtime_error("Empty slice load attempt");

    // 1. Copy CPU -> GPU Buffer
    std::memcpy(_h_compressed_buffer, _ram_storage[iz].data(), size);
    
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        _d_compressed_buffer,
        _h_compressed_buffer, 
        size, 
        cudaMemcpyHostToDevice, 
        stream
    ));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // 2. Setup ZFP for Decompression
    zfp_field_set_pointer(_zfield, (void*)wfld->mat);
    
    if (_bstream) stream_close(_bstream);
    _bstream = stream_open(_d_compressed_buffer, size);
    zfp_stream_set_bit_stream(_zstream, _bstream);

    // 3. Decompress
    zfp_decompress(_zstream, _zfield);
    
    // Ensure it finished before we use the data (if zfp is async)
    // If zfp is blocking on GPU, this might be redundant but safe.
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
}

void WavefieldPool::cleanup() {
    if (_zstream) zfp_stream_close(_zstream);
    if (_zfield)  zfp_field_free(_zfield);
    if (_bstream) stream_close(_bstream); // Check header if stream_close exists or bitstream_close
    
    if (_d_compressed_buffer) cudaFree(_d_compressed_buffer);
    if (_h_compressed_buffer) cudaFreeHost(_h_compressed_buffer);
    _ram_storage.clear();
}