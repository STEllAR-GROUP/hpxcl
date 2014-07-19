# Copyright (c)       2013 Martin Stumpf
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# set library exports
add_definitions(-DHPX_OPENCL_MODULE_EXPORTS)

# allow old opencl 1.1 calls
add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_1_APIS)

add_hpx_component(opencl
   ESSENTIAL
   SOURCES  std.cpp
            tools.cpp
            device.cpp
            event.cpp
            buffer.cpp
            program.cpp
            kernel.cpp
            server/std.cpp
            server/device.cpp
            server/event.cpp
            server/buffer.cpp
            server/program.cpp
            server/kernel.cpp
            server/hpx_cl_interop.cpp
            component_definitions.cpp
   HEADERS  fwd_declarations.hpp
            export_definitions.hpp
            std.hpp
            tools.hpp
            device.hpp
            event.hpp
            buffer.hpp
            program.hpp
            kernel.hpp
            enqueue_overloads.hpp
            server/std.hpp
            server/device.hpp
            server/event.hpp
            server/buffer.hpp
            server/program.hpp
            server/kernel.hpp
            server/hpx_cl_interop.hpp
   COMPONENT_DEPENDENCIES iostreams 
   DEPENDENCIES ${OPENCL_LIBRARIES}
   )

