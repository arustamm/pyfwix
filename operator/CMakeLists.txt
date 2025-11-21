#Requied CMake version
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

#Name of our cmake project
project(CudaOperator LANGUAGES CXX CUDA)

#Require c++111
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# enable -fPic and include the path to the cur dir
set(CMAKE_POSITION_INDEPENDENT_CODE True)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

#Required packages
find_package( Boost 1.40 COMPONENTS program_options unit_test_framework REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )

find_package(CUDAToolkit REQUIRED cufft_static cudart_static)

#include the giee directories when compiling and linking
find_package(SEPlib REQUIRED)
if(SEPlib_FOUND)
	message("-- Found SEPlib")
	include_directories(${SEPlib_INCLUDE_DIRS})
	link_directories(${SEPlib_LIBRARY_DIRS})
endif(SEPlib_FOUND)

find_package(GTest REQUIRED)

# find compression library
find_package(SZ3 REQUIRED CONFIG)
# Get include directories from the SZ3::SZ3 target
get_target_property(SZ3_INCLUDE_DIRS SZ3::SZ3 INTERFACE_INCLUDE_DIRECTORIES)

if(SZ3_INCLUDE_DIRS)
	message("-- SZ3_INCLUDE_DIRS found from target: ${SZ3_INCLUDE_DIRS}")
	include_directories(${SZ3_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "-- SZ3_INCLUDE_DIRS not found in the SZ3::SZ3 target.")
endif()


#add the rules for building the C++ code
include_directories(src)
add_subdirectory(src)
enable_testing()
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_subdirectory(tests)
# pybind wrapper
add_subdirectory(src/python)
