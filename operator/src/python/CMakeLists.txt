
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

pybind11_add_module(pyCudaOperator MODULE pyCudaOperator.cpp)

target_link_libraries(pyCudaOperator PRIVATE CudaOperator)

#install the libraries
install(TARGETS pyCudaOperator DESTINATION lib/python)

#We will wrap the python/C++ interface with a strict python
install(FILES CudaOperator.py DESTINATION lib/python)
