cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(FWIX LANGUAGES CXX CUDA)

# Common settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

# Option to use consistent pybind11 version
# Set default pybind11 version if not already defined
# Share the common pybind11 instance if requested
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG 75007dda72ad4508064c1f080394eae50d3a61ee
    UPDATE_DISCONNECTED TRUE
)
FetchContent_MakeAvailable(pybind11)

# Find common dependencies once to avoid version conflicts
find_package(Boost 1.40 COMPONENTS program_options unit_test_framework REQUIRED)
find_package(CUDAToolkit REQUIRED cufft_static cudart_static)
find_package(SEPlib REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED core)
find_package(TBB)
# find compression library
find_package(SZ3 REQUIRED CONFIG)
# Get include directories from the SZ3::SZ3 target
get_target_property(SZ3_INCLUDE_DIRS SZ3::SZ3 INTERFACE_INCLUDE_DIRECTORIES)

# Add include directories for all dependencies
include_directories(
    ${SEPlib_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
    ${SZ3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIR}
)

# Add link directories for all dependencies
link_directories(
    ${SEPlib_LIBRARY_DIRS}
    ${TBB_LIBRARIES}
)



# Build the operator project first since CudaWEM depends on it
add_subdirectory(operator) 
add_subdirectory(propagator)
