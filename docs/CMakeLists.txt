# Copyright (c)       2014 Martin Stumpf
#					  2016 Patrick Diehl
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#hpx_write_boostbook_catalog(${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml)

find_package(XSLTPROC)
#find_package(BoostAutoIndex)
find_package(BoostQuickBook)
find_package(Doxygen)


file(GLOB base_files "${hpxcl_SOURCE_DIR}/docs/*.qbk")
file(GLOB build_system_files "${hpxcl_SOURCE_DIR}/docs/manual/build_system/*.qbk")
file(GLOB_RECURSE index_files "${hpxcl_SOURCE_DIR}/docs/*.idx")

set(documentation_files
    ${manual_files} ${build_system_files} ${tutorial_files}
    ${base_files} ${index_files})
    
set(documentation_dependencies
    ${hpxcl_SOURCE_DIR}/examples/opencl/hello_world.cpp
    ${hpxcl_SOURCE_DIR}/examples/opencl/list_devices.cpp
    ${hpxcl_SOURCE_DIR}/examples/cuda/streams.cpp
    ${hpxcl_SOURCE_DIR}/examples/cuda/cuda_list_devices.cpp
    ${hpxcl_SOURCE_DIR}/examples/cuda/cuda_list_extended_devices.cpp 
    ${hpxcl_SOURCE_DIR}/examples/cuda/build_kernel.cpp
    ${hpxcl_SOURCE_DIR}/examples/cuda/build_kernel_from_file.cpp
    )

set(all_documentation_dependencies
    ${documentation_files}
    ${documentation_dependencies})
    
set(doxygen_opencl 
	${hpxcl_SOURCE_DIR}/opencl/buffer.hpp
    ${hpxcl_SOURCE_DIR}/opencl/device.hpp
    ${hpxcl_SOURCE_DIR}/opencl/kernel.hpp
    ${hpxcl_SOURCE_DIR}/opencl/program.hpp
    ${hpxcl_SOURCE_DIR}/hpxcl/opencl.hpp)
 
set(doxygen_cuda 
	${hpxcl_SOURCE_DIR}/cuda/buffer.hpp
    ${hpxcl_SOURCE_DIR}/cuda/device.hpp
    ${hpxcl_SOURCE_DIR}/cuda/get_devices.hpp
    ${hpxcl_SOURCE_DIR}/cuda/program.hpp
    ${hpxcl_SOURCE_DIR}/hpxcl/cuda.hpp)
    
set(doxygen_dependencies ${doxygen_opencl} ${doxygen_cuda})

foreach(doxygen_input ${doxygen_dependencies})
  set(doxygen_inputs "${doxygen_inputs} ${doxygen_input}")
endforeach()

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
set(doxygen_output_file "${CMAKE_CURRENT_BINARY_DIR}/hpxcl_autodoc")
set(doxygen_output_dir "${CMAKE_CURRENT_BINARY_DIR}/doxygen")
set(doxygen_example_path "${hpxcl_SOURCE_DIR}/examples/opencl/ ${hpxcl_SOURCE_DIR}/examples/cuda/")

foreach(doxygen_predef ${doxygen_definition_list})
  set(doxygen_definitions "${doxygen_definitions} \"${doxygen_predef}\"")
endforeach()

foreach(doxygen_input ${doxygen_dependencies})
  set(doxygen_inputs "${doxygen_inputs} ${doxygen_input}")
endforeach()

hpx_info("doxygen" "Creating ${CMAKE_CURRENT_BINARY_DIR}/hpxcl_autodoc.doxy")

configure_file(${PROJECT_SOURCE_DIR}/docs/autodoc.doxy.in
                ${CMAKE_CURRENT_BINARY_DIR}/hpxcl_autodoc.doxy
                @ONLY)

#############################################################################
set(all_documentation_dependencies ${all_documentation_dependencies}
    ${CMAKE_CURRENT_BINARY_DIR}/hpxcl_autodoc.xml)

# Generate Doxygen from the source code.
hpx_source_to_boostbook(hpxcl_autodoc
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

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/cmake_toolchains.qbk.in"
  "${PROJECT_SOURCE_DIR}/docs/manual/build_system/cmake_toolchains.qbk"
  @ONLY
)

set(all_documentation_dependencies ${all_documentation_dependencies} ${doxygen_dependencies})

hpx_quickbook_to_html(hpxcl
  SOURCE "${hpxcl_SOURCE_DIR}/docs/hpxcl.qbk"
  INDEX "${hpxcl_SOURCE_DIR}/docs/hpxcl.idx"
  DEPENDENCIES
    ${all_documentation_dependencies}
    "${PROJECT_SOURCE_DIR}/docs/manual/build_system/cmake_variables.qbk"
    "${PROJECT_SOURCE_DIR}/docs/manual/build_system/cmake_toolchains.qbk"
  CATALOG "${CMAKE_CURRENT_BINARY_DIR}/boostbook_catalog.xml"
  XSLTPROC_ARGS chunk.section.depth=4
                chunk.first.sections=1
                toc.section.depth=4
                toc.max.depth=4
                generate.section.toc.level=4
                ${auto_index_xsl_param}
  QUICKBOOK_ARGS ${quickbook_fixme_mode} ${auto_index_mode}
  AUTOINDEX_ARGS --internal-index
  SINGLEPAGE ${HPX_WITH_DOCUMENTATION_SINGLEPAGE})
  
  if(FOP_FOUND)

  if(BOOSTAUTOINDEX_FOUND)
    hpx_docbook_to_xslfo(hpxcl SOURCE hpxcl_auto_index.dbk
                         XSLTPROC_ARGS index.on.type=1)
  else()
    hpx_docbook_to_xslfo(hpxcl SOURCE hpxcl.dbk)
  endif()

  hpx_xslfo_to_pdf(hpxcl SOURCE hpxcl.fo)
endif()

if(FOP_FOUND)
  add_custom_target(docs
    DEPENDS hpxcl_HTML.manifest hpxcl.pdf
    SOURCES ${documentation_files})
else()
  add_custom_target(docs
    DEPENDS hpxcl_HTML.manifest
    SOURCES ${documentation_files})
endif()

set_target_properties(docs
    PROPERTIES FOLDER "Documentation/")

if(HPX_BUILD_DOCUMENTATION_SINGLE)
  add_custom_target(singlepage_docs
    DEPENDS hpxcl_singlepage_HTML.manifest
    SOURCES ${documentation_files})
  set_target_properties(singlepage_docs
      PROPERTIES FOLDER "Documentation/")
endif()
  