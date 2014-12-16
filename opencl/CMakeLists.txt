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
   SOURCES  component_definitions.cpp
            device.cpp
            get_devices.cpp
            tools.cpp
            server/get_devices.cpp
            server/device.cpp
   HEADERS  device.hpp
            get_devices.hpp
            export_definitions.hpp
            fwd_declarations.hpp
            tools.hpp
            server/device.hpp
            server/get_devices.hpp
   COMPONENT_DEPENDENCIES iostreams 
   DEPENDENCIES ${OPENCL_LIBRARIES}
   )

