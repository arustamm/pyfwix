

# compiling CUDA files into objects
set(CU_SRC 
phase_shift.cu 
selector.cu
injection.cu
reflect.cu
taper.cu
scatter.cu
imaging_condition.cu
lin_reflect.cu
)

set(CU_INC 
prop_kernels.cuh
)
# add_library(cuda_objects OBJECT ${CU_SRC} ${CU_INC})
# set_property(TARGET cuda_objects PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# compiling cpp into objects
set(CPP_SRC 
PhaseShift.cpp 
RefSampler.cpp 
PSPI.cpp 
NSPS.cpp
Injection.cpp
OneWay.cpp
Reflect.cpp
Propagator.cpp
StreamingPropagator.cpp
utils.cpp
Scatter.cpp
WavefieldPool.cpp
ImagingCondition.cpp
Scattering.cpp
ExtendedBorn.cpp
LinReflect.cpp
)

set(CPP_INC 
PhaseShift.h 
RefSampler.h 
Selector.h 
OneStep.h
Injection.h
OneWay.h
Reflect.h
Propagator.h
Taper.h
StreamingPropagator.h
utils.h
Scatter.h
WavefieldPool.h
ImagingCondition.h
Scattering.h
ExtendedBorn.h
LinReflect.h
)
# add_library(cpp_objects OBJECT ${CPP_SRC} ${CPP_INC})
# set_property(TARGET cpp_objects PROPERTY CUDA_SEPARABLE_COMPILATION ON)
include_directories(${TBB_INCLUDE_DIRS})
add_library(CudaWEM STATIC ${CU_SRC} ${CU_INC} ${CPP_SRC} ${CPP_INC})

set_property(TARGET CudaWEM PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET CudaWEM PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET CudaWEM PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(CudaWEM
						CudaOperator
						genericCpp sepVector hypercube
						jsonCpp sep3d sep
					  CUDA::cudart_static CUDA::cufft_static
						${OpenCV_LIBS} tbb
						)

install(TARGETS CudaWEM DESTINATION lib)
