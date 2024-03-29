# Copyright (c)       2015 Patrick Diehl
#                     2017 Madhavan Seshadri
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    cuda_list_devices
    cuda_list_extended_devices
    build_kernel
    build_kernel_from_file
    streams
    shared_memory
    get_device_id
    p2p_test
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
   list(APPEND HPXCL_TARGETS ${example_program})
endforeach()


if(${HPXCL_BUILD_EXAMPLE_MANDELBROT}==ON)

set(subdirs
mandelbrot
)

foreach(subdir ${subdirs})
    add_subdirectory(${subdir})
endforeach()

endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/example_kernel.cu DESTINATION ${CMAKE_SOURCE_DIR}/build/examples/cuda)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/example_shared_kernel.cu DESTINATION ${CMAKE_SOURCE_DIR}/build/examples/cuda)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/p2p_test_kernel.cu DESTINATION ${CMAKE_SOURCE_DIR}/build/examples/cuda)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/writeTest_kernel.cu DESTINATION ${CMAKE_SOURCE_DIR}/build/examples/cuda)
