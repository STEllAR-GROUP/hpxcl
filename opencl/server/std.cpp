// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "std.hpp"
#include "../tools.hpp"
#include "../device.hpp"

#include <vector>
#include <string>


///////////////////////////////////////////////////
/// HPX Registration Stuff
///
HPX_REGISTER_PLAIN_ACTION(hpx::opencl::server::get_devices_action,
                          opencl_get_devices_action);

///////////////////////////////////////////////////
/// Local functions
///
static float cl_version_to_float(std::vector<char> version_str_)
{


    try{
       
        // Make String out of char array
        std::string version_str (&version_str_[0]);
    
        // Cut away the "OpenCL " in front of the version string
        version_str = version_str.substr(7);
    
        // Cut away everything behind the version number
        version_str = version_str.substr(0, version_str.find(" "));
        
        // Parse version number
        float version_number = (float) ::atof(version_str.c_str());

        // Return the parsed version number
        return version_number;

    } catch (const std::exception & ex) {
        hpx::cerr << "Error while parsing OpenCL Version!" << hpx::endl;
        return -1.0f;
    }

}

static
std::vector<char>
get_device_info(cl_device_id id, cl_device_info info_type)
{
    
    // Declairing the cl error code variable
    cl_int err;

    // Query for size
    size_t param_size;
    err = clGetDeviceInfo((cl_device_id) id, info_type, 0, NULL, &param_size);
    cl_ensure(err, "clGetDeviceInfo()");

    // Retrieve
    std::vector<char> info(param_size);
    err = clGetDeviceInfo((cl_device_id) id, info_type, param_size, &info[0], 0);
    cl_ensure(err, "clGetDeviceInfo()");

    // Return
    return info;

}

static
std::string
get_device_info_string(cl_device_id id,
                                            cl_device_info info_type)
{

    std::vector<char> char_array = get_device_info(id, info_type);

    // Calculate length of string. Cut short if it has a 0-Termination
    // (Some queries like CL_DEVICE_NAME always return a size of 64, but 
    // contain a 0-terminated string)
    size_t length = 0;
    while(length < char_array.size())
    {
        if(char_array[length] == '\0') break;
        length++;
    }

    return std::string(char_array.begin(), char_array.begin() + length);

}


///////////////////////////////////////////////////
/// Implementations
///
std::vector<hpx::opencl::device>
hpx::opencl::server::get_devices(cl_device_type type, float min_cl_version)
{
    // Declairing the cl error code variable
    cl_int err;

    // Query for number of available platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_ensure(err, "clGetPlatformIDs()");

    // Retrieve platforms
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, &platforms[0], NULL);
    cl_ensure(err, "clGetPlatformIDs()");

    // Search on every platform
    std::vector<cl_device_id> devices;
    BOOST_FOREACH(
        const std::vector<cl_platform_id>::value_type& platform, platforms)
    {
        // Query for number of available devices
        cl_uint num_devices_on_platform;
        err = clGetDeviceIDs(platform, type, 0, NULL, &num_devices_on_platform);
        if(err == CL_DEVICE_NOT_FOUND) continue;
        cl_ensure(err, "clGetDeviceIDs()");

        // Retrieve devices
        std::vector<cl_device_id> devices_on_platform(num_devices_on_platform);
        err = clGetDeviceIDs(platform, type, num_devices_on_platform,
                             &devices_on_platform[0], NULL);
        cl_ensure(err, "clGetDeviceIDs()");

        // Add devices_on_platform to devices
        BOOST_FOREACH( const std::vector<cl_device_id>::value_type& device,
                       devices_on_platform )
        {
            // Check for required OpenCL Version 
            std::vector<char> cl_version_string
                                = get_device_info( device,
                                                   CL_DEVICE_VERSION);
            float device_cl_version = cl_version_to_float(cl_version_string);
            if(device_cl_version < min_cl_version) continue;

            // Add device to list of valid devices
            devices.push_back(device);
        }
    }

    // Return found devices.
    std::vector<hpx::opencl::device> ret;
    hpx::opencl::device newdevice(hpx::components::new_<hpx::opencl::server::device>(hpx::find_here(), (clx_device_id)devices[0]));
    ret.push_back(newdevice);
    return ret;
}



