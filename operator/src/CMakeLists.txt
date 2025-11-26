

# compiling CUDA files into objects
set(CU_SRC fft_callback.cu complex_vector.cu)
set(CU_INC fft_callback.cuh complex_vector.cuh)

# compiling cpp into objects
set(CPP_SRC complex_vector.cpp FFT.cpp python/pyCudaOperator.cpp)
set(CPP_INC StreamingOperator.h complex_vector.h CudaOperator.h FFT.h)

# link the objects together
add_library(CudaOperator OBJECT ${CPP_SRC} ${CPP_INC} ${CU_SRC} ${CU_INC})
set_property(TARGET CudaOperator PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET CudaOperator PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(CudaOperator PUBLIC
					genericCpp sepVector hypercube
					jsonCpp sep3d sep
					CUDA::cudart_static CUDA::cufft_static
					pybind11::module Python::Module
				)
