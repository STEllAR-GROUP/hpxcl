
#Require a minimum version of CMake
cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)

#This project is C++ based.
project(hpxcl CXX)

# add our cmake path
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/" ${CMAKE_MODULE_PATH} )

########################################################################
# Include hpx
#

if(HPX_ROOT)
    set(HPX_DIR "${HPX_ROOT}/lib/cmake/HPX")
endif()

#Instruct cmake to find the HPX settings
if(HPX_DIR)
    message(STATUS "Searching for HPX in \"${HPX_DIR}\" ...")
endif()

find_package(HPX REQUIRED)
set(HPX_RPATH "${CMAKE_BINARY_DIR}/lib/hpx:${HPX_RPATH}:${CMAKE_INSTALL_PREFIX}/lib/hpx")
include_directories(${HPX_INCLUDE_DIR})
link_directories(${HPX_LIBRARY_DIR})

message(STATUS "Found HPX: ${HPX_PREFIX}")

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
    find_package(CUDA REQUIRED)
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

  file(COPY ${HPX_PREFIX}/bin/hpxrun.py DESTINATION ${CMAKE_BINARY_DIR}/bin)

  hpx_option(HPX_BUILD_TESTS_BENCHMARKS BOOL "Build HPX benchmark tests (default: ON)" ON ADVANCED)
  hpx_option(HPX_BUILD_TESTS_REGRESSIONS BOOL "Build HPX regression tests (default: ON)" ON ADVANCED)
  hpx_option(HPX_BUILD_TESTS_UNIT BOOL "Build HPX unit tests (default: ON)" ON ADVANCED)

  add_hpx_pseudo_target(tests)

  enable_testing()
  include(CTest)

  include_directories(tests)
  add_subdirectory(tests)
else()
  unset(HPX_BUILD_TESTS_BENCHMARKS CACHE)
  unset(HPX_BUILD_TESTS_REGRESSIONS CACHE)
  unset(HPX_BUILD_TESTS_UNIT CACHE)
endif()


