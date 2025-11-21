#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <OneStep.h>
#include <Reflect.h>

#include <sep_reg_file.h>
#include <utils.h>
#include <ioModes.h>
#include "zfp.h"
#include <queue>

class WavefieldPool {
public:
    WavefieldPool(std::shared_ptr<hypercube> wfld_hyper, 
                std::shared_ptr<paramObj> par, std::string run_id) {
        initialize(wfld_hyper, par, run_id);
    }

    ~WavefieldPool() {
        cleanup();
    }

    // Compression/decompression interface
    void compress_slice_async(int iz, complex_vector* wfld, cudaStream_t stream, 
                              std::string& tag);
   	std::pair<int,std::future<std::shared_ptr<complex4DReg>>> decompress_slice_async(int iz, const std::string& tag);

    // Resource management
    std::shared_ptr<complex4DReg> get_wfld_buffer(int pool_idx) { return wfld_pool[pool_idx]; }
    void release_decomp_buffer(int pool_idx);
    int get_pool_size() const { return wfld_pool.size(); }
    size_t get_total_compressed_size() const { return _total_compressed_size.load(); }
    
    // Pipeline management
    void check_ready();
    void wait_to_finish();
    void clear_pipeline();

private:
    void initialize(std::shared_ptr<hypercube> wfld_hyper, std::shared_ptr<paramObj> par, std::string run_id);
    void cleanup();
    
    std::future<void> compress_slice_impl(int iz, int pool_idx, cudaEvent_t event, std::string& tag);

    // Resources
    size_t _slice_size_bytes;
    std::vector<std::shared_ptr<complex4DReg>> wfld_pool;
    std::vector<cudaEvent_t> events_pool;
    std::vector<zfp_stream*> zfp_stream_pool;
    std::vector<zfp_field*> zfp_field_pool;
    double error_bound;

    // Add dedicated decompression resources
    std::vector<std::shared_ptr<complex4DReg>> _decomp_wfld_pool;
	std::vector<zfp_stream*> _decomp_zfp_stream_pool;
	std::vector<zfp_field*> _decomp_zfp_field_pool;
    
    // Pipeline management
    std::queue<std::future<void>> compression_futures;
    std::queue<std::future<void>> decompression_futures;
    std::mutex compression_mutex;  // For thread safety
    
    // Configuration
    int nwflds_to_store;
    double rel_error_bound;
    size_t slice_size_bytes;

    std::queue<int> _decomp_free_indices_queue;
    std::mutex _decomp_pool_mutex;
    std::condition_variable _decomp_pool_cv;

    std::string _base_path;
    std::atomic<size_t> _total_compressed_size;
};
