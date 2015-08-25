
#Require a minimum version of CMake
cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)

#This project is C++ based.
project(hpxcl CXX)

# Find and include HPX settings
if(NOT HPX_ROOT)
    #  message("HPX_DIR not set, checking environment variables...")
    #  set(HPX_DIR $ENV{HPX_ROOT}/lib/cmake/hpx)
    #  if(NOT DEFINED HPX_DIR)
    #message(FATAL_ERROR "HPX_ROOT not set, unable to find HPX!")
    #  endif()
    set(HPX_ROOT /usr/local)
endif()

set(HPX_DIR "${HPX_ROOT}/lib/cmake/hpx")

# add our cmake path
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/" ${CMAKE_MODULE_PATH} )

########################################################################
# Include hpx
#

#Instruct cmake to find the HPX settings
find_package(HPX REQUIRED)
set(HPX_RPATH "${CMAKE_BINARY_DIR}/lib/hpx:${HPX_RPATH}:${CMAKE_INSTALL_PREFIX}/lib/hpx")
include_directories(${HPX_INCLUDE_DIR})
link_directories(${HPX_LIBRARY_DIR})


########################################################################
# enable resource precompiler
#
include(EmbedResources)

########################################################################
# configurations
# 
hpx_option(HPX_BUILD_EXAMPLES BOOL "Build HPX examples (default: ON)" ON ADVANCED)
hpx_option(HPX_BUILD_TESTS BOOL "Build HPX tests (default: ON)" ON ADVANCED)
hpx_option(HPX_BUILD_DOCUMENTATION BOOL "The HPX documentation toolchain is available (default OFF)." OFF)
hpx_option(HPXCL_WITH_OPENCL BOOL "Builds the OPENCL part of HPXCL (default ON)." OFF)
hpx_option(HPXCL_WITH_CUDA BOOL "Builds the CUDA part of HPXCL (default OFF)." OFF)


########################################################################
# further includes, depending on config
# 
if(HPXCL_WITH_OPENCL)
    find_package(OpenCL REQUIRED)
    include_directories(${OPENCL_INCLUDE_DIRS})
	#get_filename_component(PARENT_DIR ${OPENCL_LIBRARIES} OPENCL_LIBRARY_DIR)
    #link_directories(${OPENCL_LIBRARY_DIR})
endif()

if(HPXCL_WITH_CUDA)
    find_package(CUDA 7.0 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

if(HPX_BUILD_EXAMPLES)
    find_package(PNG)
    if(PNG_FOUND)
        include_directories(${PNG_INCLUDE_DIRS})
    endif()
endif()


#########################
# Builds
# 

# build opencl
if(HPXCL_WITH_OPENCL)
    add_subdirectory(opencl)
endif()

# build cuda
if(HPXCL_WITH_CUDA)
    add_subdirectory(cuda)
endif()


if(HPX_BUILD_EXAMPLES)
    add_hpx_pseudo_target(examples)
    include_directories(examples)
    add_subdirectory(examples)
endif()

if(HPX_BUILD_DOCUMENTATION)
  hpx_option(HPX_BUILD_DOCUMENTATION_SINGLEPAGE BOOL "The HPX documentation should be build as a single page HTML (default OFF)." OFF)
  add_subdirectory(docs)
endif()


###############################################################################

if(HPX_BUILD_TESTS)

  file(COPY ${HPX_ROOT}/bin/hpxrun.py DESTINATION ${CMAKE_BINARY_DIR}/bin)

  hpx_option(HPX_BUILD_TESTS_BENCHMARKS BOOL "Build HPX benchmark tests (default: ON)" ON ADVANCED)
  hpx_option(HPX_BUILD_TESTS_REGRESSIONS BOOL "Build HPX regression tests (default: ON)" ON ADVANCED)
  hpx_option(HPX_BUILD_TESTS_UNIT BOOL "Build HPX unit tests (default: ON)" ON ADVANCED)

  add_hpx_pseudo_target(tests)

  enable_testing()
  include(CTest)

  add_custom_command(TARGET tests POST_BUILD
    COMMAND ctest --output-on-failure)

  include_directories(tests)
  add_subdirectory(tests)
else()
  unset(HPX_BUILD_TESTS_BENCHMARKS CACHE)
  unset(HPX_BUILD_TESTS_REGRESSIONS CACHE)
  unset(HPX_BUILD_TESTS_UNIT CACHE)
endif()


