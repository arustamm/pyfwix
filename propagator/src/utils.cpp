#include "utils.h"

namespace fs = std::filesystem;

// Function to generate a random alphanumeric string
std::string generate_random_code(size_t length) {
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, sizeof(alphanum) - 2);
    
    std::string code;
    code.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        code += alphanum[dist(gen)];
    }
    
    return code;
}

// Function to get file path with timestamp and random code
std::string get_output_filepath(const std::string& filename, const std::string& extension) {
    // Get DATAPATH environment variable
    const char* datapath_env = std::getenv("DATAPATH");
    
    if (datapath_env == nullptr) 
        throw std::runtime_error("DATAPATH environment variable is not set.");

    fs::path base_path = datapath_env;
    
    // Get current time
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    
    // Format timestamp
    std::ostringstream timestamp;
    timestamp << std::put_time(&tm, "%Y%m%d_%H%M%S");
    
    // Build final filename with timestamp and random code
    std::string final_filename = 
        filename + "_" + timestamp.str() + extension;
    
    // Return complete path
    return (base_path / final_filename).string();
}
