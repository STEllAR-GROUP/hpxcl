# Copyright (c)       2017 Madhavan Seshadri
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

MESSAGE(STATUS "PNG " ${PNG_LIBRARIES})

if(PNG_FOUND)

set(sources
    mandelbrot.cpp
    kernal.cu
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../..)

source_group("Source Files" FILES ${sources})

# add example executable
add_hpx_executable(mandelbrot_cuda
                   SOURCES ${sources}
                   COMPONENT_DEPENDENCIES iostreams cuda
                   FOLDER "Examples/CUDA/mandelbrot")

# add a custom target for this example
add_hpx_pseudo_target(examples.cuda.mandelbrot_cuda)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.cuda
                            examples.cuda.mandelbrot_cuda)

target_link_libraries(mandelbrot_cuda_exe ${PNG_LIBRARIES})

endif(PNG_FOUND)


file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/kernal.cu DESTINATION ${CMAKE_SOURCE_DIR}/build/example/cuda/mandelbrot/kernel.cu)