cmake_minimum_required(VERSION 3.19)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90") 
endif()

project(fwix LANGUAGES C CXX CUDA)

# Common settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

include(FetchContent)
FetchContent_Declare(
  zfp
  GIT_REPOSITORY https://github.com/LLNL/zfp.git
  GIT_TAG        1.0.1 
)

# FORCE ZFP options before fetching
set(ZFP_WITH_CUDA ON CACHE BOOL "" FORCE)
set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} CACHE STRING "" FORCE)
set(ZFP_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(zfp)

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
    ${OpenCV_LIBS} TBB::tbb
    pybind11::module Python::Module
)

set (PY_INSTALL 
fwix/__init__.py 
fwix/CudaOperator.py 
fwix/CudaWEM.py 
fwix/FWIXmodeling.py 
fwix/FWIXProblem.py 
fwix/utils.py 
fwix/workers.py
fwix/sep3d.py
)

install(TARGETS pyFWIX DESTINATION fwix)
install(FILES ${PY_INSTALL} DESTINATION fwix)