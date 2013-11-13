// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "std.hpp"
#include "../tools.hpp"

#include <vector>

using hpx::opencl::clx_device_id;


///////////////////////////////////////////////////
/// HPX Registration Stuff
///
HPX_REGISTER_PLAIN_ACTION(hpx::opencl::server::clGetDeviceIDs_action,
                          clGetDeviceIDs_action);
HPX_REGISTER_PLAIN_ACTION(hpx::opencl::server::clGetDeviceInfo_action,
                          clGetDeviceInfo_action);


///////////////////////////////////////////////////
/// Implementations
///
std::vector<clx_device_id>
hpx::opencl::server::clGetDeviceIDs(cl_device_type type)
{
    // Declairing the cl error code variable
    cl_int err;

    // Query for number of available platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_ensure(err, "clGetPlatformIDs");

    // Retrieve platforms
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, &platforms[0], NULL);
    cl_ensure(err, "clGetPlatformIDs");

    // Search on every platform
    std::vector<clx_device_id> devices;
    BOOST_FOREACH(
        const std::vector<cl_platform_id>::value_type& platform, platforms)
    {
        // Query for number of available devices
        cl_uint num_devices_on_platform;
        err = clGetDeviceIDs(platform, type, 0, NULL, &num_devices_on_platform);
        if(err == CL_DEVICE_NOT_FOUND) continue;
        cl_ensure(err, "clGetDeviceIDs");

        // Retrieve devices
        std::vector<cl_device_id> devices_on_platform(num_devices_on_platform);
        err = clGetDeviceIDs(platform, type, num_devices_on_platform,
                             &devices_on_platform[0], NULL);
        cl_ensure(err, "clGetDeviceIDs");

        // Add devices_on_platform to devices
        BOOST_FOREACH( const std::vector<cl_device_id>::value_type& device,
                       devices_on_platform )
        {
            devices.push_back((clx_device_id)device);
        }
    }

    // Return found devices.
    return devices;
}


std::vector<char>
hpx::opencl::server::clGetDeviceInfo(clx_device_id id, cl_device_info info_type)
{
    
    // Declairing the cl error code variable
    cl_int err;

    // Query for size
    size_t param_size;
    err = clGetDeviceInfo((cl_device_id) id, info_type, 0, NULL, &param_size);
    cl_ensure(err, "clGetDeviceInfo");

    // Retrieve
    std::vector<char> info(param_size);
    err = clGetDeviceInfo((cl_device_id) id, info_type, param_size, &info[0], 0);
    cl_ensure(err, "clGetDeviceInfo");

    // Return
    return info;

}
