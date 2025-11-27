
cmake_minimum_required(VERSION 3.25)

set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

#require python3
find_package(PythonInterp 3 REQUIRED)

#need pybind11
find_package(pybind11 REQUIRED)

#need python include directory
include_directories(${PYTHON_INCLUDE_DIR})

#create pybind11 application
include_directories(${pybind11_INCLUDE_DIR})

python_add_library(pyCudaWEM MODULE pyCudaWEM.cpp)
set_target_properties(pyCudaWEM PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON 
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Static
    LINKER_LANGUAGE CUDA  
)

target_link_libraries(pyCudaWEM PRIVATE CudaWEM)

#install the libraries
install(TARGETS pyCudaWEM DESTINATION fwix)

#We will wrap the python/C++ interface with a strict python
install(FILES CudaWEM.py DESTINATION fwix)
