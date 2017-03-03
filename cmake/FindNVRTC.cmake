# Copyright (c)    2015 Patrick Diehl
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt
##############################################################################
# - Try to find the Cuda NVRTC library
# Once done this will define
# LIBNVRTC_FOUND - System has LibNVRTC
# LIBNVRTC_LIBRARY_DIR - The NVRTC library dir
# CUDA_NVRTC_LIB - The NVRTC lib
##############################################################################
find_package(PkgConfig)

find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBNVRTC_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibNVRTC DEFAULT_MSG CUDA_NVRTC_LIB)

mark_as_advanced(CUDA_NVRTC_LIB)

if(NOT LIBNVRTC_FOUND)
message(FATAL_ERROR "Cuda NVRTC Library not found: Specify the LIBNVRTC_LIBRARY_DIR where libnvrtc is located") 
endif()
