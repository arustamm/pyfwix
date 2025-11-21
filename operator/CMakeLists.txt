#add the rules for building the C++ code
include_directories(src)
add_subdirectory(src)
enable_testing()
add_subdirectory(tests)
# pybind wrapper
add_subdirectory(src/python)
