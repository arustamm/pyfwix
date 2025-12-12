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
                std::shared_ptr<paramObj> par, std::string run_id, int max_depth) {
        initialize(wfld_hyper, par, run_id, max_depth);
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
    void initialize(std::shared_ptr<hypercube> wfld_hyper, std::shared_ptr<paramObj> par, std::string run_id, int max_depth);
    void cleanup();
    
    std::future<void> compress_slice_impl(int iz, int pool_idx, cudaEvent_t event, std::string& tag);
    int get_or_open_fd(const std::string& tag);

    // Resources
    size_t _slice_size_bytes;
    std::vector<std::shared_ptr<complex4DReg>> wfld_pool;
    std::vector<cudaEvent_t> events_pool;
    std::vector<zfp_stream*> zfp_stream_pool;
    std::vector<zfp_field*> zfp_field_pool;
    double compress_rate;
    
    int _fd = -1;
    int _max_depth = 0;
    size_t _chunk_size = 0;
    std::map<std::string, int> _fd_map;
    std::mutex _io_mutex; // Protects the map opening
    
    std::vector<std::vector<char>> _compressed_buffer_pool; 
    std::vector<std::vector<char>> _decomp_buffer_pool;

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

    std::queue<int> _comp_free_indices_queue;
    std::mutex _comp_pool_mutex;
    std::condition_variable _comp_pool_cv;
    
    // For tracking ALL in-flight compressions
    std::queue<std::future<void>> _comp_futures_queue;
    std::mutex _comp_queue_mutex;

    std::string _base_path;
    std::atomic<size_t> _total_compressed_size;
};