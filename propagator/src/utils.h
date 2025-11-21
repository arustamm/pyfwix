#pragma once
#include <cstdlib>      // for getenv
#include <filesystem>   // for path operations (C++17)
#include <fstream>      // for file operations
#include <random>       // for random number generation
#include <ctime>        // for time
#include <iomanip>      // for setw, setfill
#include <sstream>      // for stringstream

namespace fs = std::filesystem;

// Function to generate a random alphanumeric string
std::string generate_random_code(size_t length = 8);

// Function to get file path with timestamp and random code
std::string get_output_filepath(const std::string& filename, const std::string& extension = ".H");