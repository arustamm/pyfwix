

# compiling CUDA files into objects
set(CU_SRC fft_callback.cu complex_vector.cu)
set(CU_INC fft_callback.cuh complex_vector.cuh)
# add_library(cuda_objects OBJECT ${CU_SRC} ${CU_INC})
# set_property(TARGET cuda_objects PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# compiling cpp into objects
set(CPP_SRC complex_vector.cpp FFT.cpp)
set(CPP_INC StreamingOperator.h complex_vector.h CudaOperator.h FFT.h)
# add_library(cpp_objects OBJECT ${CPP_SRC} ${CPP_INC})
# set_property(TARGET cpp_objects PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# link the objects together
add_library(CudaOperator STATIC ${CPP_SRC} ${CPP_INC} ${CU_SRC} ${CU_INC})
set_property(TARGET CudaOperator PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET CudaOperator PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET CudaOperator PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(CudaOperator
						genericCpp sepVector hypercube
						jsonCpp sep3d sep
					  CUDA::cudart_static CUDA::cufft_static
						)

install(TARGETS CudaOperator DESTINATION lib)
