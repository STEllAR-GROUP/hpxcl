
#Require a minimum version of CMake
cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)

#This project is C++ based.
project(hpxcl CXX)

# This adds the HPX cmake configuration directory to the search path.

message(STATUS
${HPX_ROOT}/share/cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}/Modules)
set(CMAKE_MODULE_PATH
${HPX_ROOT}/share/cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}/Modules)

# add our cmake path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

###########################
# Include boost and hpx
#
find_package(HPX_Boost)

hpx_include_sys_directories("${BOOST_INCLUDE_DIR}")
hpx_link_sys_directories("${BOOST_LIBRARY_DIR}")


#Instruct cmake to find the HPX settings
find_package(HPX REQUIRED)
set(HPX_RPATH "${CMAKE_BINARY_DIR}/lib/hpx:${HPX_RPATH}:${CMAKE_INSTALL_PREFIX}/lib/hpx")
include_directories(${HPX_INCLUDE_DIR})
link_directories(${HPX_LIBRARY_DIR})

########################################################################
# configurations
# 
hpx_option(HPX_BUILD_EXAMPLES BOOL "Build HPX examples (default: ON)" ON ADVANCED)
hpx_option(HPX_BUILD_TESTS BOOL "Build HPX tests (default: ON)" ON ADVANCED)
hpx_option(HPX_BUILD_DOCUMENTATION BOOL "The HPX documentation toolchain is available (default OFF)." OFF)
hpx_option(HPXCL_WITH_OPENCL BOOL "Builds the CUDA part of HPXCL (default ON)." ON)
hpx_option(HPXCL_WITH_CUDA BOOL "Builds the CUDA part of HPXCL (default OFF)." OFF)


########################################################################
# further includes, depending on config
# 
if(HPXCL_WITH_OPENCL)
    find_package(HPX_OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIR})
    link_directories(${OpenCL_LIBRARY_DIR})
endif()

if(HPXCL_WITH_CUDA)
    find_package(CUDA REQUIRED)
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


