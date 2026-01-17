#pragma once
#include <CudaOperator.h>
#include <complex4DReg.h>
#include <paramObj.h>
#include <zfp.h>
#include <vector>
#include <mutex>
#include <cstring> // for std::memcpy

class WavefieldPool {
public:
    WavefieldPool(std::shared_ptr<hypercube> wfld_hyper, 
                std::shared_ptr<paramObj> par, std::string run_id, int max_depth) {
        initialize(wfld_hyper, par, run_id, max_depth);
    }

    ~WavefieldPool() {
        cleanup();
    }

    // -------------------------------------------------------
    // Interface: Save/Load directly to/from RAM
    // -------------------------------------------------------
    
    // Compresses the GPU buffer 'wfld' and saves it to RAM at index 'iz'
    void save_slice(int iz, complex_vector* wfld, cudaStream_t stream);

    // Loads from RAM at index 'iz', decompresses into GPU buffer 'wfld'
    void load_slice(int iz, complex_vector* wfld, cudaStream_t stream);

    size_t get_compressed_size() {
        size_t total_size = 0;
        for (const auto& slice : _ram_storage) {
            total_size += slice.size() * sizeof(unsigned char);
        }
        return total_size;
    }

private:
    void initialize(std::shared_ptr<hypercube> wfld_hyper, std::shared_ptr<paramObj> par, std::string run_id, int max_depth);
    void cleanup();

    // -------------------------------------------------------
    // In-Memory Storage
    // -------------------------------------------------------
    // _ram_storage[iz] holds the compressed binary data for depth iz
    std::vector<std::vector<unsigned char>> _ram_storage;
    std::vector<float> _error_bounds; // Store error bounds per slice
    
    // -------------------------------------------------------
    // cuSZp Resources
    // -------------------------------------------------------
    uint3 _dims;              // The "Folded" dimensions
    double _rel_error;       // Compression tolerance
    size_t _nbEle;            // Total number of float elements
    size_t _max_comp_bytes;   // Allocation size for buffers

    // GPU Output Buffer (Reusable temporary storage)
    unsigned char* _d_compressed_buffer; 
    
    // Host Staging Buffer (Pinned memory for fast PCIe transfer)
    unsigned char* _h_compressed_buffer; 

    // Compressed Size Tracker (Pinned Memory)
    // This matches 'cmpSize1' in the example, but allocated as pinned
    // so the GPU can write to it and CPU can read it.
    size_t* _cmpSize; 

    zfp_stream* _zstream = nullptr;
    zfp_field* _zfield  = nullptr;
    bitstream* _bstream = nullptr;
    
    size_t _compressed_size_bytes = 0;
};