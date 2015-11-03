# Copyright (c)       2014 Martin Stumpf
#                     2014 Damond Howard
#                     2015 Patrick Diehl
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

add_definitions(-DHPX_CUDA_MODULE_EXPORTS)
include_directories(${PROJECT_SOURCE_DIR})

add_hpx_component(cuda
    AUTOGLOB
    ESSENTIAL
    COMPONENT_DEPENDENCIES iostreams
    DEPENDENCIES ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_NVRTC_LIB}
    FOLDER "Components/CUDA")

# Add installation of the header files
install(FILES
    ${CMAKE_SOURCE_DIR}/hpx_cuda.hpp
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/cuda/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/include/cuda
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN "CMakeFiles" EXCLUDE)

