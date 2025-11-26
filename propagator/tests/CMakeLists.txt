
include_directories(${TBB_INCLUDE_DIRS})
# compiling cpp into objects

set(TEST_SOURCES 
prop_unit_test.cpp 
# pspi_benchmark.cpp
prop_benchmark.cpp
stream_unit_test.cpp
stream_benchmark.cpp
ps_benchmark.cpp
born_unit_test.cpp
born_benchmark.cpp
)

foreach(src ${TEST_SOURCES})
    string(REPLACE ".cpp" "" obj ${src})
    add_executable(${obj} ${src})
    target_sources(${obj} PRIVATE 
        $<TARGET_OBJECTS:CudaOperator>
        $<TARGET_OBJECTS:CudaWEM>
    )
    set_property(TARGET ${obj} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(${obj}
                                genericCpp sepVector hypercube
                                jsonCpp sep3d sep
                                CUDA::cudart_static CUDA::cufft_static zfp::zfp
                                ${OpenCV_LIBS} tbb
                                pybind11::module Python::Module
                                GTest::gtest_main GTest::gtest
                            )
    add_test(NAME ${obj} COMMAND ${obj})
endforeach()

