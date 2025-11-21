#include <memory>
#include <vector>
#include <complex>
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>

// Include necessary headers for your classes
#include "hypercube.h"
#include "axis.h"
#include "complex4DReg.h"
#include "PhaseShift.h"

// Helper to create hypercube
std::shared_ptr<hypercube> createHypercube(const std::vector<axis>& axes) {
    return std::make_shared<hypercube>(axes);
}

// Helper to format bytes to human-readable form
std::string formatBytes(size_t bytes) {
    const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    int suffix_idx = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024 && suffix_idx < 4) {
        size /= 1024;
        suffix_idx++;
    }
    
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << suffixes[suffix_idx];
    return ss.str();
}

int main(int argc, char **argv) {
    // Fixed dimensions
    const int nx = 1000;
    const int ny = 400;
    
    // Default values
    float dz = 10.0f;
    float eps = 0.01f;
    int num_iterations = 3; // Small number for quick testing
    
    // Default nw values to try if none provided
    std::vector<int> nw_values;
    
    // Parse command-line arguments - multiple nw values
    if (argc >= 2) {
        for (int i = 1; i < argc; i++) {
            try {
                int nw = std::stoi(argv[i]);
                if (nw > 0) {
                    nw_values.push_back(nw);
                }
            } catch (...) {
                // Skip invalid values
            }
        }
    }
    
    // If no valid values provided, use defaults
    if (nw_values.empty()) {
        nw_values = {50, 100, 200};
        std::cout << "No valid nw values provided. Using defaults: 50, 100, 200" << std::endl;
    }
    
    std::cout << "PhaseShift Benchmark - nx=" << nx << ", ny=" << ny << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;
    std::cout << std::setw(8) << "nw" 
              << std::setw(16) << "Memory Size" 
              << std::setw(16) << "Forward (GB/s)" 
              << std::setw(20) << "Forward (ps/elem)" 
              << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;
    
    // Size of a complex<float> in bytes
    const size_t complex_size = sizeof(std::complex<float>);
    
    // Run benchmark for each nw value
    for (int nw : nw_values) {
        // Create axes and hypercube
        std::vector<axis> axes = {
            axis(nx, 0.0f, 10.0f),
            axis(ny, 0.0f, 10.0f),
            axis(nw, 0.0f, 30.0f),
            axis(1, 0.0f, 1.0f)
        };
        
        auto domain = createHypercube(axes);
        size_t num_elements = domain->getN123();
        
        // Calculate memory size in bytes
        size_t memory_bytes = num_elements * complex_size;
        
        // Create model and data
        auto model = std::make_shared<complex4DReg>(domain);
        auto data = std::make_shared<complex4DReg>(domain);
        model->set(1.0f);
        
        // Create PhaseShift operator
        PhaseShift phaseShift(domain, dz);
        
        // Warm-up run
        phaseShift.forward(false, model, data);
        cudaDeviceSynchronize();
        
        // Benchmark forward
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; i++) {
            phaseShift.cu_forward(false, phaseShift.model_vec, phaseShift.data_vec);
            cudaDeviceSynchronize();
        }
        
        
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calculate timings in different units
        double forward_sec = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 
                           (static_cast<double>(num_iterations) * 1000000.0);
        
        double forward_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 
                           static_cast<double>(num_iterations);
        
        // Calculate throughputs
        // GB/s - (read model + write data) / time
        double forward_throughput_GBs = (memory_bytes * 2) / (forward_sec * 1e9);
        
        // Time per element (ns/element)
        double forward_ns_per_element = forward_ns * 1000 / num_elements;
        
        // Print forward results
        std::cout << std::setw(8) << nw 
                  << std::setw(16) << formatBytes(memory_bytes)
                  << std::setw(16) << std::fixed << std::setprecision(2) << forward_throughput_GBs 
                  << std::setw(20) << std::fixed << std::setprecision(2) << forward_ns_per_element 
                  << std::endl;
        
    }
    
    
    return 0;
}