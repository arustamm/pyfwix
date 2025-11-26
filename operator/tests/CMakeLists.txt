

# compiling cpp into objects

set(TEST_SOURCES 
operator_unit_test.cpp 
fft_benchmark.cpp 
complex_vector_unit_test.cpp
)

    foreach(src ${TEST_SOURCES})
        string(REPLACE ".cpp" "" obj ${src})
        add_executable(${obj} ${src})
        target_sources(${obj} PRIVATE 
            $<TARGET_OBJECTS:CudaOperator>
        )
        set_property(TARGET ${obj} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        target_link_libraries(${obj}
					genericCpp sepVector hypercube
					jsonCpp sep3d sep
					CUDA::cudart_static CUDA::cufft_static zfp::zfp
					GTest::gtest_main GTest::gtest fftw3f fftw3f_threads -lm)
        add_test(NAME ${obj} COMMAND ${obj})
    endforeach()

add_test(cuOp_gtests operator_unit_test)
