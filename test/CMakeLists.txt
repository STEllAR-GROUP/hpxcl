
#Require a minimum version of CMake
cmake_minimum_required(VERSION 2.8.4 FATAL_ERROR)

#This project is C++ based.
project(simple_test CXX)

# This adds the HPX cmake configuration directory to the search path.

message(STATUS
${HPX_ROOT}/share/cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}/Modules)
set(CMAKE_MODULE_PATH
${HPX_ROOT}/share/cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}/Modules)

#Instruct cmake to find the HPX settings
find_package(HPX REQUIRED)
set(HPX_RPATH "${CMAKE_INSTALL_PREFIX}/lib/hpx:${HPX_RPATH}")
include_directories(${HPX_INCLUDE_DIR})
link_directories(${HPX_LIBRARY_DIR})

find_package(HPX_OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIR})
link_directories(${OpenCL_LIBRARY_DIR})


add_subdirectory(hpxcl/opencl)

#build your application using HPX
add_hpx_executable(opencl_client
   ESSENTIAL
   SOURCES opencl_client.cpp
   COMPONENT_DEPENDENCIES opencl_components
   DEPENDENCIES opencl_std_lib
   )
