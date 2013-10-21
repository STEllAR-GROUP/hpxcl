add_hpx_library(opencl_std
   ESSENTIAL
   SOURCES std.cpp server/std.cpp
   HEADERS std.hpp server/std.hpp name_definitions.hpp
   COMPONENT_DEPENDENCIES iostreams
   DEPENDENCIES ${OpenCL_LIBRARY} opencl_tools_lib
   )

add_hpx_library(opencl_tools
   ESSENTIAL
   SOURCES tools.cpp
   HEADERS tools.hpp
   DEPENDENCIES ${OpenCL_LIBRARY}
   )


add_hpx_component(opencl_device
   ESSENTIAL
   SOURCES device.cpp server/device.cpp
   HEADERS device.hpp server/device.hpp
   COMPONENT_DEPENDENCIES iostreams
   DEPENDENCIES ${OpenCL_LIBRARY} opencl_tools_lib
   )

add_hpx_component(opencl_memory
   ESSENTIAL
   SOURCES memory.cpp server/memory.cpp
   HEADERS memory.hpp server/memory.hpp
   COMPONENT_DEPENDENCIES opencl_device
   DEPENDENCIES ${OpenCL_LIBRARY} opencl_tools_lib
   )

add_hpx_component(opencl_buffer
    ESSENTIAL
    SOURCES buffer.cpp server/buffer.cpp
    HEADERS buffer.hpp server/buffer.hpp
    COMPONENT_DEPENDENCIES opencl_memory
    DEPENDENCIES ${OpenCL_LIBRARY} opencl_tools_lib
    )
