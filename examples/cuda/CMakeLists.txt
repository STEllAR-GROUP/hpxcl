# Copyright (c)       2015 Patrick Diehl
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    cuda_list_devices
    cuda_list_extended_devices
    build_kernel
    build_kernel_from_file
    streams
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

endforeach()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu DESTINATION ../../examples/cuda)
