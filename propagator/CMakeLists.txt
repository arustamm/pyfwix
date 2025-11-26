# my Operator library
include_directories(../operator/src)
# link_directories(../operator/lib)

#add the rules for building the C++ code
include_directories(src)
add_subdirectory(src)
enable_testing()

add_subdirectory(tests)
# add_subdirectory(src/python)

# link_directories(${PROJECT_SOURCE_DIR}/lib)
# add_subdirectory(test)
