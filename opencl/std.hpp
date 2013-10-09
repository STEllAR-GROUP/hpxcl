// Copyright (c)	2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_STD_HPP__
#define HPX_OPENCL_STD_HPP__

#include <hpx/include/iostreams.hpp>

#include <CL/cl.h>

#include <vector>

#include "server/std.hpp"
#include "name_definitions.hpp"

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{
	
    // Get all devices on node
	std::vector<clx_device_id> clGetDeviceIDs(hpx::naming::id_type,
                                              cl_device_type);	
    // Get device information
	void clGetDeviceInfo( hpx::naming::id_type,
                          clx_device_id,
                          cl_device_info,
                          size_t param_value_size,
                          void *param_value,
                          size_t *param_value_size_ret);	

}}



#endif
