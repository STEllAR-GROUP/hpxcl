# Copyright (c) 2015 Patrick Diehl
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

############################################################################
# Configure HPXCL benchmarks
############################################################################

set(example_programs
   StreamHPX
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

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cu DESTINATION ${CMAKE_SOURCE_DIR}/build/benchmark/cuda/stream/)
