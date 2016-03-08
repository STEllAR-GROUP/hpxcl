# Copyright (c) 2014 Martin Stumpf
# Copyright (c) 2015 Patrick Diehl
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Require a minimum version of CMake
cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Configuration type (one of Debug, RelWithDebInfo, Release, MinSizeRel)")
endif()

# This project is C++ based.
project(hpxcl CXX)

# add our cmake path
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

if(HPX_ROOT)
    set(HPX_DIR "${HPX_ROOT}/lib/cmake/HPX")
endif()

#add -std=gnu++11 to the debug flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	if (CMAKE_COMPILER_IS_GNUCC)
		 SET(CMAKE_CXX_FLAGS_DEBUG "-ggdb -D_DEBUG -O0 -std=gnu++11" CACHE STRING)
	endif()
endif() 

########################################################################
# Include hpx
# Instruct cmake to find the HPX settings
find_package(HPX REQUIRED)

if(NOT (HPX_BUILD_TYPE STREQUAL CMAKE_BUILD_TYPE))
    message("HPX_BUILD_TYPE: " ${HPX_BUILD_TYPE} "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
    message(FATAL_ERROR "Build types used for HPX and this library have to match")
endif()

# inherit varios setting from HPXConfig
set(CMAKE_DEBUG_POSTFIX ${HPX_DEBUG_POSTFIX})

# Adjust auxiliary build settings for HPX to work properly
if(NOT MSVC)
    set(HPX_RPATH
        "${PROJECT_BINARY_DIR}/lib/hpx:${HPX_RPATH}:${CMAKE_INSTALL_PREFIX}/lib/hpx")
else()
    # enable solution folders for MSVC
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    if(HPX_NO_INSTALL)
      set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/$(Configuration)" CACHE PATH
          "Prefix prepended to install directories." FORCE)
    endif()
endif()


list(APPEND CMAKE_MODULE_PATH "${HPX_PREFIX}/../cmake/")
include_directories(${HPX_INCLUDE_DIR})
link_directories(${HPX_LIBRARY_DIR})

########################################################################
# enable resource precompiler
#
include(EmbedResources)

########################################################################
# configurations
hpx_option(HPXCL_BUILD_EXAMPLES BOOL "Build HPX examples (default: ON)" ON ADVANCED)
hpx_option(HPXCL_BUILD_TESTS BOOL "Build HPX tests (default: ON)" ON ADVANCED)
hpx_option(HPXCL_WITH_DOCUMENTATION BOOL "The HPX documentation toolchain is available (default OFF)." OFF)
hpx_option(HPXCL_WITH_OPENCL BOOL "Builds the OPENCL part of HPXCL (default ON)." OFF)
hpx_option(HPXCL_WITH_CUDA BOOL "Builds the CUDA part of HPXCL (default OFF)." OFF)
hpx_option(HPXCL_BUILD_BENCHMARK BOOL "Builds the benchmark (default OFF)." OFF)

if(NOT HPXCL_WITH_OPENCL AND NOT HPXCL_WITH_CUDA)
    message(FATAL_ERROR "At least one of HPXCL_WITH_OPENCL or HPXCL_WITH_CUDA has to be set to ON")
endif()

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
    find_package(NVRTC 7.0 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

#Only search for PNG lib, if HPXCL_WITH_OPENCL is build
if(HPXCL_BUILD_EXAMPLES AND HPXCL_WITH_OPENCL)
    find_package(PNG)
    if(PNG_FOUND)
        include_directories(${PNG_INCLUDE_DIRS})
    endif()
endif()

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

if(HPXCL_BUILD_EXAMPLES)
    add_hpx_pseudo_target(examples)
    include_directories(examples)
    add_subdirectory(examples)
endif()

if(HPXCL_WITH_DOCUMENTATION)
  include(HPX_Documentation)
  hpx_option(HPX_BUILD_DOCUMENTATION_SINGLEPAGE BOOL "The HPX documentation should be build as a single page HTML (default OFF)." OFF)
  add_subdirectory(docs)
endif()

if(HPXCL_BUILD_BENCHMARK)
	add_hpx_pseudo_target(benchmark)
	include_directories(benchmark)
    add_subdirectory(benchmark)
endif()

###############################################################################

#if(HPXCL_BUILD_TESTS)
#
#  file(COPY ${HPX_ROOT}/bin/hpxrun.py DESTINATION ${PROJECT_BINARY_DIR}/bin)
#
#  hpx_option(HPXCL_BUILD_TESTS_BENCHMARKS BOOL "Build HPX benchmark tests (default: ON)" ON ADVANCED)
#  hpx_option(HPXCL_BUILD_TESTS_REGRESSIONS BOOL "Build HPX regression tests (default: ON)" ON ADVANCED)
#  hpx_option(HPXCL_BUILD_TESTS_UNIT BOOL "Build HPX unit tests (default: ON)" ON ADVANCED)
#
#  add_hpx_pseudo_target(tests)
#
#  enable_testing()
#  include(CTest)
#
#  add_custom_command(TARGET tests POST_BUILD
#    COMMAND ctest --output-on-failure)
#
#  include_directories(tests)
#  add_subdirectory(tests)
#else()
  unset(HPXCL_BUILD_TESTS_BENCHMARKS CACHE)
  unset(HPXCL_BUILD_TESTS_REGRESSIONS CACHE)
  unset(HPXCL_BUILD_TESTS_UNIT CACHE)
#endif()


