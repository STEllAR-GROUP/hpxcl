# Copyright (c) 2015 Patrick Diehl
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPXCL_WITH_NAIVE_CUDA_BENCHMARK)
	############################################################################
	# Configure pure CUDA benchmarks
	############################################################################
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
	SET(CUDA_NVCC_FLAGS "-use_fast_math;  
	-gencode arch=compute_30,code=sm_30;
	-gencode arch=compute_35,code=sm_35; -gencode arch=compute_50,code=sm_50;
	-gencode arch=compute_52,code=sm_52; -gencode arch=compute_52,code=compute_52; -std=c++11" 
	CACHE "NVCC_FLAGS" STRING FORCE) 

	#Benchmark for single stencil kernel
	CUDA_ADD_EXECUTABLE(StencilCuda StencilCUDA.cu)
	#Benchmark for partitions
	CUDA_ADD_EXECUTABLE(PartitionCuda PartitionCUDA.cu)
	#Benchmark for partitions using multiple devices
	CUDA_ADD_EXECUTABLE(PartitionCudaMultiDevice PartitionCUDAMultiDevice.cu)
endif()

############################################################################
# Configure HPXCL benchmarks
############################################################################

set(example_programs
    StencilHPX
    PartitionHPX
    PartitionHPXMultiDevice
)

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  include_directories(${CMAKE_SOURCE_DIR})
  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     COMPONENT_DEPENDENCIES iostreams cuda
                     DEPENDENCIES ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY}
                     FOLDER "Benchmark/CUDA/${example_program}")

endforeach()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/stencil_kernel.cu DESTINATION ${CMAKE_SOURCE_DIR}/build/benchmark/cuda/stencil/)
