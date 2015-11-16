# Copyright (c) 2015 Patrick Diehl
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

############################################################################
# Configure pure CUDA benchmarks
############################################################################
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(CUDA_NVCC_FLAGS "-use_fast_math; -gencode arch=compute_20,code=sm_20; 
-gencode arch=compute_20,code=sm_21; -gencode arch=compute_30,code=sm_30;
-gencode arch=compute_35,code=sm_35; -gencode arch=compute_50,code=sm_50;
-gencode arch=compute_52,code=sm_52; -gencode arch=compute_52,code=compute_52; -std=c++11" 
CACHE "NVCC_FLAGS" STRING FORCE) 

#Benchmark without streams
CUDA_ADD_EXECUTABLE(BenchmarkCuda BenchmarkCUDA.cu)

############################################################################
# Configure HPXCP benchmarks
############################################################################

set(example_programs
    BenchmarkHPX
)

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)
  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     COMPONENT_DEPENDENCIES iostreams cuda
                     DEPENDENCIES ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY}
                     FOLDER "Examples/CUDA/${example_program}")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.cuda.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.cuda
                              examples.cuda.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.cuda.${example_program}
                              ${example_program}_exe)
                              
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu DESTINATION ../../../examples/cuda/benchmark_vector)

endforeach()