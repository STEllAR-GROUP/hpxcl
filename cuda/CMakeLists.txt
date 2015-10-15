# Copyright (c)       2014 Martin Stumpf
#                     2014 Damond Howard
#                     2015 Patrick Diehl
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

cuda_add_library(cuda_kernels cuda/kernel.cu)

add_definitions(-DHPX_CUDA_MODULE_EXPORTS)
include_directories(${PROJECT_SOURCE_DIR})

add_hpx_component(libhpx_cuda
    AUTOGLOB
    ESSENTIAL
    COMPONENT_DEPENDENCIES iostreams)

target_link_libraries(libhpx_cuda_component
    ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_NVRTC_LIB})
