#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <WavefieldPool.h>
#include <ioModes.h>
#include "zfp.h"
#include <queue>
#include <iomanip>      // For std::setw and std::setfill
#include <filesystem>   // For std::filesystem::create_directory (C++17)
#include <fcntl.h>  // open
#include <unistd.h> // pread, pwrite, close, lseek

void WavefieldPool::initialize(
	std::shared_ptr<hypercube> wfld_hyper, 
	std::shared_ptr<paramObj> par,
	std::string run_id,
	int max_depth
) { 
	
	_slice_size_bytes = wfld_hyper->getN123() * sizeof(std::complex<float>);

	int nwflds_to_store = par->getInt("wflds_to_store", 3);
	compress_rate = par->getFloat("compress_rate", 8.);
	std::string base_wfld_dir = par->getString("wfld_path", "/tmp");
	_base_path = base_wfld_dir + "/" + run_id; 
	std::filesystem::create_directory(_base_path);

	// Create shared pools
	_max_depth = max_depth;
	wfld_pool.resize(nwflds_to_store);
	events_pool.resize(nwflds_to_store);
	zfp_stream_pool.resize(nwflds_to_store);
	zfp_field_pool.resize(nwflds_to_store);
	_decomp_wfld_pool.resize(nwflds_to_store);
	_decomp_zfp_stream_pool.resize(nwflds_to_store);
	_decomp_zfp_field_pool.resize(nwflds_to_store);

	_compressed_buffer_pool.resize(nwflds_to_store);
    _decomp_buffer_pool.resize(nwflds_to_store);

	auto ax = wfld_hyper->getAxes();
	
	// Initialize shared resources...
	for (int i = 0; i < nwflds_to_store; ++i) {
		wfld_pool[i] = std::make_shared<complex4DReg>(wfld_hyper);
		CHECK_CUDA_ERROR(cudaHostRegister(wfld_pool[i]->getVals(), 
						_slice_size_bytes, 
						cudaHostRegisterDefault));
		CHECK_CUDA_ERROR(cudaEventCreate(&events_pool[i]));
		// Initialize ZFP resources...
		zfp_stream_pool[i] = zfp_stream_open(NULL);
		zfp_stream_set_rate(zfp_stream_pool[i], compress_rate, zfp_type_float, 4, 0);
		// Note: The data pointer is null, it will be set just-in-time
		zfp_field_pool[i] = zfp_field_4d(nullptr, zfp_type_float, 2*ax[0].n, ax[1].n, ax[2].n, ax[3].n);

		// Decompress resources
		_decomp_wfld_pool[i] = std::make_shared<complex4DReg>(wfld_hyper);
		CHECK_CUDA_ERROR(cudaHostRegister(_decomp_wfld_pool[i]->getVals(), _slice_size_bytes, cudaHostRegisterDefault));
		_decomp_zfp_stream_pool[i] = zfp_stream_open(NULL);
		zfp_stream_set_rate(_decomp_zfp_stream_pool[i], compress_rate, zfp_type_float, 4, 0);
		// We associate the zfp_field with the buffer pointer just-in-time during decompression
		_decomp_zfp_field_pool[i] = zfp_field_4d(nullptr, zfp_type_float, 2*ax[0].n, ax[1].n, ax[2].n, ax[3].n);

		// Add the index to the queue of free buffers
		_decomp_free_indices_queue.push(i);
		_comp_free_indices_queue.push(i); 
		_total_compressed_size = 0; // Initialize the counter

		if (i == 0) {
            // Calculate size ONCE based on the first stream
            _chunk_size = zfp_stream_maximum_size(zfp_stream_pool[0], zfp_field_pool[0]);
        }
		_compressed_buffer_pool[i].resize(_chunk_size);
        _decomp_buffer_pool[i].resize(_chunk_size);
	}
}

int WavefieldPool::get_or_open_fd(const std::string& tag) {
    std::lock_guard<std::mutex> lock(_io_mutex); // Protect map access
    
    if (_fd_map.find(tag) != _fd_map.end()) {
        return _fd_map[tag];
    }

    // File doesn't exist yet for this tag. Create it.
    std::string filename = _base_path + "/checkpoints_" + tag + ".bin";
    
    // O_TRUNC ensures we don't read garbage from previous runs
    int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0664); 
    if (fd == -1) throw std::runtime_error("Failed to open file: " + filename);
    
    // Pre-allocate for performance
    size_t total_size = (size_t)_max_depth * _chunk_size;
    #ifdef __linux__
    posix_fallocate(fd, 0, total_size);
    #endif

    _fd_map[tag] = fd;
    return fd;
}

void WavefieldPool::compress_slice_async(int iz, complex_vector* __restrict__ wfld, cudaStream_t stream, std::string& tag) {
	int pool_idx;
    {
        std::unique_lock<std::mutex> lock(_comp_pool_mutex);
        _comp_pool_cv.wait(lock, [this]{ return !_comp_free_indices_queue.empty(); });
        pool_idx = _comp_free_indices_queue.front();
        _comp_free_indices_queue.pop();
    }
	auto event = events_pool[pool_idx];
	auto wfld_buffer = wfld_pool[pool_idx];
	
	// Copy GPU data to host buffer
	CHECK_CUDA_ERROR(cudaMemcpyAsync(wfld_buffer->getVals(), wfld->mat, 
					_slice_size_bytes, cudaMemcpyDeviceToHost, stream));
	
	// Record event after copy
	CHECK_CUDA_ERROR(cudaEventRecord(event, stream));
	
	// Launch async compression
	auto future = compress_slice_impl(iz, pool_idx, event, tag);
	
	{
        std::lock_guard<std::mutex> lock(_comp_queue_mutex);
        _comp_futures_queue.push(std::move(future));
    }
	
}

std::future<void> WavefieldPool::compress_slice_impl(int iz, int pool_idx, cudaEvent_t event, std::string& tag) {
	return std::async(std::launch::async, [this, iz, pool_idx, event, &tag]() {

		int fd = get_or_open_fd(tag);

		auto wfld_buffer = wfld_pool[pool_idx];
		auto zfp_s = zfp_stream_pool[pool_idx];
		auto zfp_f = zfp_field_pool[pool_idx];

		// Wait for GPU->CPU copy to complete
		CHECK_CUDA_ERROR(cudaEventSynchronize(event));

		// Set up ZFP field
		zfp_field_set_pointer(zfp_f, reinterpret_cast<float*>(wfld_buffer->getVals()));

		// Compress
		char* buffer_ptr = _compressed_buffer_pool[pool_idx].data();
        size_t buffer_cap = _compressed_buffer_pool[pool_idx].size();
		
		bitstream* stream = stream_open(buffer_ptr, buffer_cap);
		zfp_stream_set_bit_stream(zfp_s, stream);
		zfp_stream_rewind(zfp_s);

		size_t size = zfp_compress(zfp_s, zfp_f);
		stream_close(stream);

		if (size == 0) {
			throw std::runtime_error("Compression failed for slice " + std::to_string(iz));
		}
		_total_compressed_size += size;

		// 3. ATOMIC WRITE (pwrite)
        // Write 'size' bytes to the specific offset for this depth 'iz'
        off_t offset = (off_t)iz * _chunk_size;
        ssize_t ret = pwrite(fd, buffer_ptr, size, offset);
        
        if (ret != size) throw std::runtime_error("Disk Write failed at iz: " + std::to_string(iz));

		{
            std::lock_guard<std::mutex> lock(_comp_pool_mutex);
            _comp_free_indices_queue.push(pool_idx);
        }
        _comp_pool_cv.notify_one();
	});
}

std::pair<int,std::future<std::shared_ptr<complex4DReg>>> WavefieldPool::decompress_slice_async(int iz, const std::string& tag) {

// Acquire a free buffer from the pool
	int pool_idx;
	{
		std::unique_lock<std::mutex> lock(_decomp_pool_mutex);
		// Wait until the queue is not empty
		_decomp_pool_cv.wait(lock, [this]{ return !_decomp_free_indices_queue.empty(); });
		
		pool_idx = _decomp_free_indices_queue.front();
		_decomp_free_indices_queue.pop();
	} 

	auto future = std::async(std::launch::async, [this, iz, pool_idx, &tag]() {
			
		int fd = get_or_open_fd(tag);
		auto wfld_buffer = _decomp_wfld_pool[pool_idx];
		auto zfp_s = _decomp_zfp_stream_pool[pool_idx];
		auto zfp_f = _decomp_zfp_field_pool[pool_idx];

		// 2. ATOMIC READ (pread)
		char* buffer_ptr = _decomp_buffer_pool[pool_idx].data();
        off_t offset = (off_t)iz * _chunk_size;
        ssize_t ret = pread(fd, buffer_ptr, _chunk_size, offset);
        if (ret != _chunk_size) throw std::runtime_error("Disk Read failed at iz: " + std::to_string(iz));
		
		// Decompression logic
		bitstream* stream = stream_open((void*)buffer_ptr, _chunk_size);
		zfp_stream_set_bit_stream(zfp_s, stream);
		zfp_stream_rewind(zfp_s);

		zfp_field_set_pointer(zfp_f, reinterpret_cast<float*>(wfld_buffer->getVals()));

		if (!zfp_decompress(zfp_s, zfp_f)) {
				stream_close(stream);
				throw std::runtime_error("ZFP decompression failed for slice " + std::to_string(iz));
		}
		stream_close(stream);
		
		return wfld_buffer;
	});

	return {pool_idx, std::move(future)};
}

void WavefieldPool::release_decomp_buffer(int pool_idx) {
{
	std::lock_guard<std::mutex> lock(_decomp_pool_mutex);
	_decomp_free_indices_queue.push(pool_idx);
}
// Notify one waiting thread that a buffer is now available
_decomp_pool_cv.notify_one();
}

// Clean up completed futures (non-blocking)
void WavefieldPool::check_ready() {
    std::lock_guard<std::mutex> lock(_comp_queue_mutex);
    while (!_comp_futures_queue.empty()) {
        auto& fut = _comp_futures_queue.front();
        if (fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            fut.get();  // Propagate exceptions
            _comp_futures_queue.pop();
        } else {
            break;  // Stop at first incomplete task
        }
    }
}

void WavefieldPool::wait_to_finish() {
    std::lock_guard<std::mutex> lock(_comp_queue_mutex);
    while (!_comp_futures_queue.empty()) {
        _comp_futures_queue.front().wait();
        _comp_futures_queue.pop();
    }
}

void WavefieldPool::cleanup() {
    wait_to_finish();
    
    // Close file descriptor
    for (auto const& [tag, fd] : _fd_map) {
        if (fd != -1) close(fd);
    }
    _fd_map.clear();

    // Standard CUDA cleanup
    for (const auto& wfld_ptr : wfld_pool) {
        if (wfld_ptr && wfld_ptr->getVals()) CHECK_CUDA_ERROR(cudaHostUnregister(wfld_ptr->getVals()));
    }
    for (auto event : events_pool) if (event) CHECK_CUDA_ERROR(cudaEventDestroy(event));
    
    // Free ZFP structs
    for (size_t i = 0; i < zfp_stream_pool.size(); ++i) {
        if (zfp_stream_pool[i]) zfp_stream_close(zfp_stream_pool[i]);
        if (zfp_field_pool[i]) zfp_field_free(zfp_field_pool[i]);
    }
    
    // Decomp cleanup...
    for (const auto& wfld_ptr : _decomp_wfld_pool) {
        if (wfld_ptr && wfld_ptr->getVals()) CHECK_CUDA_ERROR(cudaHostUnregister(wfld_ptr->getVals()));
    }
    for (size_t i = 0; i < _decomp_zfp_stream_pool.size(); ++i) {
        if (_decomp_zfp_stream_pool[i]) zfp_stream_close(_decomp_zfp_stream_pool[i]);
        if (_decomp_zfp_field_pool[i]) zfp_field_free(_decomp_zfp_field_pool[i]);
    }

    // Optional: Delete the file after run
    std::filesystem::remove_all(_base_path);
}