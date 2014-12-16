// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "device.hpp"

#include "../tools.hpp"

#include <CL/cl.h>

using namespace hpx::opencl::server;


// Constructor
device::device()
{


}

// Destructor
device::~device()
{


}

// Initialization function.
// Needed because cl_device_id can not be serialized.
void
device::init(cl_device_id _device_id, bool enable_profiling)
{

    hpx::cout << std::string("Threadsize: ")
              << std::hex << hpx::threads::get_ctx_ptr()->get_stacksize()
              << std::string("\n") << hpx::flush;
    this->device_id = _device_id;

}


hpx::util::serialize_buffer<char>
device::get_device_info(cl_device_info info_type)
{
    
    // Declairing the cl error code variable
    cl_int err;

    // Query for size
    size_t param_size;
    err = clGetDeviceInfo(device_id, info_type, 0, NULL, &param_size);
    cl_ensure(err, "clGetDeviceInfo()");

    // Retrieve
    hpx::util::serialize_buffer<char> info( new char[param_size],
                                            param_size,
                                            hpx::util::serialize_buffer<char>::take);
    err = clGetDeviceInfo(device_id, info_type, param_size, info.data(), 0);
    cl_ensure(err, "clGetDeviceInfo()");

    // Return
    return info;

}


hpx::util::serialize_buffer<char>
device::get_platform_info(cl_platform_info info_type)
{
    
    // Declairing the cl error code variable
    cl_int err;

    // Query for size
    size_t param_size;
    err = clGetPlatformInfo(platform_id, info_type, 0, NULL, &param_size);
    cl_ensure(err, "clGetPlatformInfo()");

    // Retrieve
    hpx::util::serialize_buffer<char> info( new char[param_size],
                                            param_size,
                                            hpx::util::serialize_buffer<char>::take);
    err = clGetPlatformInfo(platform_id, info_type, param_size, info.data(), 0);
    cl_ensure(err, "clGetPlatformInfo()");

    // Return
    return info;

}
