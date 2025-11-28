cmake_minimum_required(VERSION 3.19)
project(fwix LANGUAGES C CXX CUDA)

# Common settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

# Find common dependencies once to avoid version conflicts
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Boost REQUIRED)
include_directories( ${Boost_INCLUDE_DIR} )
find_package(TBB REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3f fftw3)

find_package(CUDAToolkit REQUIRED cufft cudart)

find_package(SEPlib REQUIRED HINTS "$ENV{HOME}/.local/sep-io-fwix/cmake/SEP")
if(SEPlib_FOUND)
	message("-- Found SEPlib")
	include_directories(${SEPlib_INCLUDE_DIRS})
	link_directories(${SEPlib_LIBRARY_DIRS})
endif(SEPlib_FOUND)

find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED core )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(zfp REQUIRED CONFIG)
message(STATUS "Found ZFP: ${zfp_DIR}")

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

# Build the operator project first since CudaWEM depends on it
add_subdirectory(operator) 
add_subdirectory(propagator)
pybind11_add_module(pyFWIX MODULE
    fwix/src/pyFWIX.cpp
    $<TARGET_OBJECTS:CudaOperator>
    $<TARGET_OBJECTS:CudaWEM>
)
set_target_properties(pyFWIX PROPERTIES 
    # 1. Force CMake to use 'nvcc' for linking this target
    LINKER_LANGUAGE CUDA
    
    # 2. Trigger the Device Link step
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(pyFWIX PRIVATE
    genericCpp sepVector hypercube
    jsonCpp sep3d sep
    CUDA::cudart_static CUDA::cufft_static zfp::zfp
    ${OpenCV_LIBS} tbb
    pybind11::module Python::Module
)
install(TARGETS pyFWIX DESTINATION fwix)
install(FILES fwix/__init__.py DESTINATION fwix)
install(FILES fwix/CudaOperator.py DESTINATION fwix)
install(FILES fwix/CudaWEM.py DESTINATION fwix)
install(FILES fwix/FWIXmodeling.py DESTINATION fwix)