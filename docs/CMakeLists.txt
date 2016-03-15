# Copyright (c)       2014 Martin Stumpf
#					  2016 Patrick Diehl
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#hpx_write_boostbook_catalog(${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml)

find_package(XSLTPROC)
find_package(Doxygen)

file(GLOB base_files ${hpxcl_SOURCE_DIR}/docs/*.qbk)
file(GLOB_RECURSE manual_files ${hpxcl_SOURCE_DIR}/docs/manual/*.qbk)
file(GLOB_RECURSE tutorial_files ${hpxcl_SOURCE_DIR}/docs/tutorial/*.qbk)
file(GLOB_RECURSE index_files ${hpxcl_SOURCE_DIR}/docs/*.idx)

set(documentation_files
    ${manual_files} ${tutorial_files} ${base_files} ${index_files})

set(documentation_dependencies
    ${hpxcl_SOURCE_DIR}/examples/opencl/hello_world.cpp
    ${hpxcl_SOURCE_DIR}/examples/cuda/streams.cpp
    ${hpxcl_SOURCE_DIR}/examples/cuda/cuda_list_devices.cpp
    ${hpxcl_SOURCE_DIR}/examples/cuda/cuda_list_extended_devices.cpp 
    ${hpxcl_SOURCE_DIR}/examples/cuda/build_kernel.cpp
    ${hpxcl_SOURCE_DIR}/examples/cuda/build_kernel_from_file.cpp
    )

set(all_documentation_dependencies
    ${documentation_files}
    ${documentation_dependencies})

# Add all files here which should be passed to doxygen

set(doxygen_opencl 
	${hpxcl_SOURCE_DIR}/opencl/buffer.hpp
    ${hpxcl_SOURCE_DIR}/opencl/device.hpp
    ${hpxcl_SOURCE_DIR}/opencl/kernel.hpp
    ${hpxcl_SOURCE_DIR}/opencl/event.hpp
    ${hpxcl_SOURCE_DIR}/opencl/program.hpp)



set(doxygen_cuda 
	${hpxcl_SOURCE_DIR}/cuda/buffer.hpp
    ${hpxcl_SOURCE_DIR}/cuda/device.hpp
    ${hpxcl_SOURCE_DIR}/cuda/get_devices.hpp
    ${hpxcl_SOURCE_DIR}/cuda/program.hpp)

set(doxygen_dependencies ${doxygen_opencl} ${doxygen_cuda})

foreach(doxygen_input ${doxygen_dependencies})
  set(doxygen_inputs "${doxygen_inputs} ${doxygen_input}")
endforeach()

set(doxygen_output_file "${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc")
set(doxygen_output_dir "${CMAKE_CURRENT_BINARY_DIR}/doxygen")

set(doxygen_definition_list
    "DOXYGEN:=1"
    "BOOST_SCOPED_ENUM_START(x):=enum x"
    "BOOST_SCOPED_ENUM_END="
    "BOOST_SYSTEM_NOEXCEPT="
    "HPX_EXCEPTION_EXPORT="
    "HPX_EXPORT="
    "HPX_API_EXPORT="
    "HPX_STD_FUNCTION=std::function"
    "extern=")

#############################################################################
# Definitions for autodoc.doxy.in
set(doxygen_output_file "${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc")
set(doxygen_output_dir "${CMAKE_CURRENT_BINARY_DIR}/doxygen")

foreach(doxygen_predef ${doxygen_definition_list})
  set(doxygen_definitions "${doxygen_definitions} \"${doxygen_predef}\"")
endforeach()

foreach(doxygen_input ${doxygen_dependencies})
  set(doxygen_inputs "${doxygen_inputs} ${doxygen_input}")
endforeach()

hpx_info("doxygen" "Creating ${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc.doxy")

configure_file(${PROJECT_SOURCE_DIR}/docs/autodoc.doxy.in
                ${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc.doxy
                @ONLY)

#############################################################################
set(all_documentation_dependencies ${all_documentation_dependencies}
    ${CMAKE_CURRENT_BINARY_DIR}/hpx_autodoc.xml)

# Generate Doxygen from the source code.
hpx_source_to_boostbook(hpx_autodoc
    CATALOG ${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml
    DEPENDENCIES ${doxygen_dependencies})

set(quickbook_fixme_mode "-D__hpx_disable_fixme__")
if(HPX_DOCUMENTATION_DEVELOPMENT)
  set(quickbook_fixme_mode "-D__hpx_enable_fixme__")
endif()

if(BOOSTAUTOINDEX_FOUND)
  set(auto_index_mode "-D__use_auto_index__")
  set(auto_index_xsl_param --stringparam index.on.type 1)
endif()

#############################################################################
add_custom_target(documentation doxygen hpx_autodoc.doxy )
