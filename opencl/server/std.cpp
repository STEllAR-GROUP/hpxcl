// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "std.hpp"
#include "../tools.hpp"
#include "../device.hpp"

#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/static.hpp>

#include <vector>
#include <string>

///////////////////////////////////////////////////
/// STATIC STUFF
///

// This variable will hold the device list on every node
using hpx::lcos::local::spinlock;
static hpx::util::static_<std::vector<hpx::opencl::device>> devices;
static hpx::util::static_<spinlock> devices_lock;

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


///////////////////////////////////////////////////
/// Implementations
///

// This method initializes the devices-list if it's not done yet.
void
ensure_device_components_initialization()
{

    boost::lock_guard<spinlock> lock(devices_lock.get());
    // Don't initialize if already initialized
    if(devices.get().size() != 0)
        return;

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
    BOOST_FOREACH(
        const std::vector<cl_platform_id>::value_type& platform, platforms)
    {
        // Query for number of available devices
        cl_uint num_devices_on_platform;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL,
                                                     &num_devices_on_platform);
        if(err == CL_DEVICE_NOT_FOUND) continue;
        cl_ensure(err, "clGetDeviceIDs()");

        // Retrieve devices
        std::vector<cl_device_id> devices_on_platform(num_devices_on_platform);
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                                  num_devices_on_platform,
                                  &devices_on_platform[0], NULL);
        cl_ensure(err, "clGetDeviceIDs()");

        // Add devices_on_platform to devices
        BOOST_FOREACH( const std::vector<cl_device_id>::value_type& device,
                       devices_on_platform )
        {

            // Create a new device client 
            hpx::opencl::device device_client(
                hpx::components::new_<hpx::opencl::server::device>(
                            hpx::find_here(),
                            (hpx::opencl::server::clx_device_id)device
                                                    ));

            // Add device to list of valid devices
            devices.get().push_back(device_client);
        }
    }

}
    

std::vector<hpx::opencl::device>
hpx::opencl::server::get_devices(cl_device_type type, float min_cl_version)
{

    // Create the list of device clients
    ensure_device_components_initialization();

    // Lock the list
    boost::lock_guard<spinlock> lock(devices_lock.get());

    // Generate a list of suitable devices
    std::vector<hpx::opencl::device> suitable_devices;
    BOOST_FOREACH( const std::vector<hpx::opencl::device>::value_type& device,
                   devices.get())
    {
        // Check for required opencl version
        std::vector<char> cl_version_string =
                                device.get_device_info(CL_DEVICE_VERSION).get();
        float device_cl_version = cl_version_to_float(cl_version_string);
        if(device_cl_version < min_cl_version) continue;

        // Check for requested device type
        std::vector<char> device_type_string = 
                                   device.get_device_info(CL_DEVICE_TYPE).get();
        cl_device_type device_type = *((cl_device_type*)
                                                     (&device_type_string[0]));
        if(!(device_type & type)) continue;

        // TODO filter devices
        suitable_devices.push_back(device);
    }

    // Return the devices found
    return suitable_devices;

}



