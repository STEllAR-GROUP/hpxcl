# Copyright (c)       2013 Martin Stumpf
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# set library exports
add_definitions(-DHPX_OPENCL_MODULE_EXPORTS)

# allow old opencl 1.1 calls
add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_1_APIS)

if(MSVC)
  hpx_add_compile_flag(-Ox CONFIGURATIONS Release LANGUAGES CXX C)
  hpx_add_compile_flag(-RTC1 CONFIGURATIONS Debug LANGUAGES CXX C)

  # VS2012 and above has a special flag for improving the debug experience by
  # adding more symbol information to the build (-d2Zi)
  hpx_add_compile_flag(-d2Zi+ CONFIGURATIONS RelWithDebInfo)

  # VS2013 and above know how to do link time constant data segment folding
  # VS2013 update 2 and above know how to remove debug information for
  #     non-referenced functions and data (-Zc:inline)
  if(MSVC12 OR MSVC13)
    hpx_add_compile_flag(-Zc:inline)
    hpx_add_compile_flag(-Gw CONFIGURATIONS Release RelWithDebInfo MinSizeRelease)
    hpx_add_compile_flag(-Zo CONFIGURATIONS RelWithDebInfo)
  endif()
  hpx_add_compile_flag(-EHsc)       # Exceptions
  hpx_add_compile_flag(-GR)         # Runtime type information
  hpx_add_compile_flag(-MP)         # Multiprocessor build
  hpx_add_compile_flag(-bigobj)     # Increase the maximum size of object file sections
endif()

add_hpx_component(opencl
   ESSENTIAL
   SOURCES  component_definitions.cpp
            device.cpp
            get_devices.cpp
            tools.cpp
            buffer.cpp
            util/generic_buffer.cpp
            util/enqueue_overloads.cpp
            lcos/event.cpp
            server/device.cpp
            server/buffer.cpp
            server/get_devices.cpp
            server/util/event_map.cpp
            server/util/event_dependencies.cpp
   HEADERS  device.hpp
            get_devices.hpp
            export_definitions.hpp
            fwd_declarations.hpp
            tools.hpp
            buffer.hpp
            cl_headers.hpp
            util/generic_buffer.hpp
            util/enqueue_overloads.hpp
            lcos/event.hpp
            lcos/future.hpp
            detail/info_type.hpp
            server/device.hpp
            server/buffer.hpp
            server/get_devices.hpp
            server/util/event_map.hpp
            server/util/event_dependencies.hpp
   COMPONENT_DEPENDENCIES iostreams
   DEPENDENCIES ${OPENCL_LIBRARIES}
   )

