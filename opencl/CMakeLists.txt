#add_hpx_library(opencl_event
#    ESSENTIAL
#    SOURCES clx_event.cpp
#    HEADERS clx_event.hpp
#    DEPENDENCIES ${OpenCL_LIBRARY}
#)

add_hpx_library(opencl_tools
   ESSENTIAL
   SOURCES tools.cpp
   HEADERS tools.hpp
   DEPENDENCIES ${OpenCL_LIBRARY}
   )

add_hpx_library(opencl_std
   ESSENTIAL
   SOURCES std.cpp server/std.cpp
   HEADERS std.hpp server/std.hpp name_definitions.hpp
   COMPONENT_DEPENDENCIES iostreams
   DEPENDENCIES ${OpenCL_LIBRARY} opencl_tools_lib
   )


add_hpx_component(opencl_components
   ESSENTIAL
   SOURCES  device.cpp
            buffer.cpp
            server/device.cpp
            server/event.cpp
            server/buffer.cpp
   HEADERS  device.hpp
            event.hpp
            buffer.hpp
            server/device.hpp
            server/event.hpp
            server/buffer.hpp
   COMPONENT_DEPENDENCIES iostreams 
   DEPENDENCIES ${OpenCL_LIBRARY} opencl_tools_lib opencl_std_lib
   )

