# Copyright (c)       2014 Martin Stumpf
#                     2014 Damond Howard
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cuda_add_library(cuda_kernels cuda/kernel.cu)


add_hpx_component(cuda
   ESSENTIAL
   SOURCES  #buffer.cpp
            device.cpp
            server/device.cpp
            #event.cpp
            kernel.cpp
            server/kernel.cpp
   HEADERS  fwd_declarations.hpp
            buffer.hpp
            device.hpp
            event.hpp
            kernel.hpp
            cuda/kernel.cuh
            server/buffer.hpp
            server/device.hpp
            server/event.hpp
            server/kernel.hpp
            stubs/buffer.hpp
            stubs/device.hpp
            stubs/event.hpp
            stubs/kernel.hpp
   COMPONENT_DEPENDENCIES iostreams 
   DEPENDENCIES cuda cudart cuda_kernels
   )

