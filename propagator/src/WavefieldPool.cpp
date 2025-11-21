#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <WavefieldPool.h>
#include <ioModes.h>
#include "zfp.h"
#include <queue>
#include <iomanip>      // For std::setw and std::setfill
#include <filesystem>   // For std::filesystem::create_directory (C++17)

void WavefieldPool::initialize(
	std::shared_ptr<hypercube> wfld_hyper, 
	std::shared_ptr<paramObj> par,
	std::string run_id) { 
	
	_slice_size_bytes = wfld_hyper->getN123() * sizeof(std::complex<float>);

	int nwflds_to_store = par->getInt("wflds_to_store", 3);
	error_bound = par->getFloat("compress_error", 1E-6);
	std::string base_wfld_dir = par->getString("wfld_path", "/tmp");
	_base_path = base_wfld_dir + "/" + run_id; 
	std::filesystem::create_directory(_base_path);

	// Create shared pools
	wfld_pool.resize(nwflds_to_store);
	events_pool.resize(nwflds_to_store);
	zfp_stream_pool.resize(nwflds_to_store);
	zfp_field_pool.resize(nwflds_to_store);
	_decomp_wfld_pool.resize(nwflds_to_store);
	_decomp_zfp_stream_pool.resize(nwflds_to_store);
	_decomp_zfp_field_pool.resize(nwflds_to_store);

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
	if (error_bound > 0) {
		zfp_stream_set_accuracy(zfp_stream_pool[i], error_bound);
	}
	else {
		zfp_stream_set_reversible(zfp_stream_pool[i]);
	}
	// Note: The data pointer is null, it will be set just-in-time
	zfp_field_pool[i] = zfp_field_4d(nullptr, zfp_type_float, 2*ax[0].n, ax[1].n, ax[2].n, ax[3].n);

	// Decompress resources
	_decomp_wfld_pool[i] = std::make_shared<complex4DReg>(wfld_hyper);
	CHECK_CUDA_ERROR(cudaHostRegister(_decomp_wfld_pool[i]->getVals(), _slice_size_bytes, cudaHostRegisterDefault));
	_decomp_zfp_stream_pool[i] = zfp_stream_open(NULL);
	if (error_bound > 0) {
		zfp_stream_set_accuracy(_decomp_zfp_stream_pool[i], error_bound);
	}
	else {
		zfp_stream_set_reversible(_decomp_zfp_stream_pool[i]);
	}
	// We associate the zfp_field with the buffer pointer just-in-time during decompression
	_decomp_zfp_field_pool[i] = zfp_field_4d(nullptr, zfp_type_float, 2*ax[0].n, ax[1].n, ax[2].n, ax[3].n);

	// Add the index to the queue of free buffers
	_decomp_free_indices_queue.push(i);
	_total_compressed_size = 0; // Initialize the counter
	}
}

void WavefieldPool::compress_slice_async(int iz, complex_vector* __restrict__ wfld, cudaStream_t stream, std::string& tag) {
	int pool_idx = iz % wfld_pool.size();
	auto event = events_pool[pool_idx];
	auto wfld_buffer = wfld_pool[pool_idx];
	
	// Copy GPU data to host buffer
	CHECK_CUDA_ERROR(cudaMemcpyAsync(wfld_buffer->getVals(), wfld->mat, 
					_slice_size_bytes, cudaMemcpyDeviceToHost, stream));
	
	// Record event after copy
	CHECK_CUDA_ERROR(cudaEventRecord(event, stream));
	
	// Launch async compression
	auto future = compress_slice_impl(iz, pool_idx, event, tag);
	
		// std::lock_guard<std::mutex> lock(compression_mutex);
	compression_futures.push(std::move(future));
	// }
	
}

std::future<void> WavefieldPool::compress_slice_impl(int iz, int pool_idx, cudaEvent_t event, std::string& tag) {
	return std::async(std::launch::async, [this, iz, pool_idx, event, &tag]() {
		auto wfld_buffer = wfld_pool[pool_idx];
		auto zfp_s = zfp_stream_pool[pool_idx];
		auto zfp_f = zfp_field_pool[pool_idx];

		// Wait for GPU->CPU copy to complete
		CHECK_CUDA_ERROR(cudaEventSynchronize(event));

		// Set up ZFP field
		zfp_field_set_pointer(zfp_f, reinterpret_cast<float*>(wfld_buffer->getVals()));

		// Compress
		size_t max_size = zfp_stream_maximum_size(zfp_s, zfp_f);
		std::vector<char> compressed_data(max_size);
		
		bitstream* stream = stream_open(compressed_data.data(), compressed_data.size());
		zfp_stream_set_bit_stream(zfp_s, stream);
		zfp_stream_rewind(zfp_s);

		size_t actual_size = zfp_compress(zfp_s, zfp_f);
		stream_close(stream);

		if (actual_size == 0) {
			throw std::runtime_error("Compression failed for slice " + std::to_string(iz));
		}

		compressed_data.resize(actual_size);
		_total_compressed_size += actual_size;
		// Thread-safe storage
		// {
			// std::lock_guard<std::mutex> lock(compression_mutex);
		// storage[iz] = std::move(compressed_data);
		// }
		// 4. Write the compressed data to the unique file
		std::string filename = _base_path + "/" + tag + "_iz_" + std::to_string(iz) + ".zfp";
		std::ofstream outfile(filename, std::ios::binary);
		if (!outfile.is_open()) 
			throw std::runtime_error("Failed to open file for writing: " + filename);
		
		outfile.write(compressed_data.data(), actual_size);
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
			
		auto wfld_buffer = _decomp_wfld_pool[pool_idx];
		auto zfp_s = _decomp_zfp_stream_pool[pool_idx];
		auto zfp_f = _decomp_zfp_field_pool[pool_idx];

		std::string filename = _base_path + "/" + tag + "_iz_" + std::to_string(iz) + ".zfp";
		// Read the file containing the compressed data
		std::ifstream infile(filename, std::ios::binary);
		if (!infile.is_open()) 
				throw std::runtime_error("Failed to open file for reading: " + filename);
		// Read the entire file into a vector
		std::vector<char> compressed_data(
			(std::istreambuf_iterator<char>(infile)),
			std::istreambuf_iterator<char>()
		);

		if (compressed_data.empty()) 
				throw std::runtime_error("Error: Compressed data for slice " + std::to_string(iz) + " is empty.");
		
		// Decompression logic
		bitstream* stream = stream_open((void*)compressed_data.data(), compressed_data.size());
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

void WavefieldPool::check_ready() {
	// std::lock_guard<std::mutex> lock(compression_mutex);
	if (compression_futures.size() >= wfld_pool.size()) {
		compression_futures.front().wait();
		compression_futures.pop();
	}
}

void WavefieldPool::wait_to_finish() {
	// std::lock_guard<std::mutex> lock(compression_mutex);
	while (!compression_futures.empty()) {
		compression_futures.front().wait();
		compression_futures.pop();
	}
}

void WavefieldPool::cleanup() {
	while (!compression_futures.empty()) {
		compression_futures.front().wait();
		compression_futures.pop();
	}

	for (const auto& wfld_ptr : wfld_pool) {
		if (wfld_ptr && wfld_ptr->getVals()) {
			// This is the counterpart to cudaHostRegister
			CHECK_CUDA_ERROR(cudaHostUnregister(wfld_ptr->getVals()));
		}
	}

	for (auto event : events_pool) {
		if (event) {
			CHECK_CUDA_ERROR(cudaEventDestroy(event));
		}
	}

	for (size_t i = 0; i < zfp_stream_pool.size(); ++i) {
		if (zfp_stream_pool[i]) {
			zfp_stream_close(zfp_stream_pool[i]);
		}
		if (zfp_field_pool[i]) {
			zfp_field_free(zfp_field_pool[i]);
		}
	}

	// --- Cleanup Decompression Resources (NEW) ---
	for (const auto& wfld_ptr : _decomp_wfld_pool) {
		if (wfld_ptr && wfld_ptr->getVals()) {
			CHECK_CUDA_ERROR(cudaHostUnregister(wfld_ptr->getVals()));
		}
	}
	for (size_t i = 0; i < _decomp_zfp_stream_pool.size(); ++i) {
		if (_decomp_zfp_stream_pool[i]) zfp_stream_close(_decomp_zfp_stream_pool[i]);
		if (_decomp_zfp_field_pool[i]) zfp_field_free(_decomp_zfp_field_pool[i]);
	}

	try {
		// Check if the path exists before trying to remove it
		if (std::filesystem::exists(_base_path)) {
			// Recursively remove the directory and all its contents.
			// This function is powerful and will delete everything inside the path.
			std::filesystem::remove_all(_base_path);
		}
	} catch (const std::filesystem::filesystem_error& e) {
		// Report an error but don't throw an exception from the destructor
		std::cerr << "WavefieldPool Destructor Error: Could not delete directory '"
							<< _base_path << "'. Reason: " << e.what() << std::endl;
	}
}



