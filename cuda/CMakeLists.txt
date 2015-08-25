# Copyright (c)       2014 Martin Stumpf
#                     2014 Damond Howard
#                     2015 Patrick Diehl
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
cuda_add_library(cuda_kernels cuda/kernel.cu)


add_hpx_component(cuda
   ESSENTIAL
   SOURCES  get_devices.cpp
            device.cpp
            buffer.cpp
            kernel.cpp
            program.cpp
            server/program.cpp  
            server/get_devices.cpp 
            server/kernel.cpp
            server/buffer.cpp
            server/device.cpp
   HEADERS  fwd_declarations.hpp
            get_devices.hpp
            server/get_devices.hpp
            buffer.hpp
            device.hpp
            kernel.hpp
            program.hpp
            tools.hpp
            cuda/kernel.cuh
            server/buffer.hpp
            server/device.hpp
            server/event.hpp
            server/kernel.hpp
            server/program.hpp
   COMPONENT_DEPENDENCIES iostreams 
   DEPENDENCIES cuda cudart cuda_kernels
   )

