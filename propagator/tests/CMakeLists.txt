
include_directories(${TBB_INCLUDE_DIRS})
# compiling cpp into objects

set(TEST_SOURCES 
prop_unit_test.cpp 
pspi_benchmark.cpp
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
    target_link_libraries(${obj} CudaWEM CudaOperator
                                genericCpp sepVector hypercube
                                CUDA::cudart_static CUDA::cufft_static
                                GTest::gtest_main GTest::gtest
                                tbb benchmark)
    add_test(NAME ${obj} COMMAND ${obj})
endforeach()

