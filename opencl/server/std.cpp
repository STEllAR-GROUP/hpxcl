// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "std.hpp"
#include "../tools.hpp"
#include "../device.hpp"

#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/static.hpp>
#include <hpx/runtime.hpp>

#include <vector>
#include <string>

///////////////////////////////////////////////////
/// STATIC STUFF
///

using hpx::lcos::local::spinlock;

// serves as a unique tag to get the device 
struct global_device_list_tag {};

// Will be set to true once the device list got initialized.
static bool device_list_initialized = false;
// Will be set to true once the device shutdown hook is set.
static bool device_shutdown_hook_initialized = false;

// This defines a static device list type.
// Generating instances of this type will always give the same list.
typedef
hpx::util::static_<std::vector<hpx::opencl::device>,
                   global_device_list_tag>  static_device_list_type;

// This defines a static device list lock type.
// Generating instances of this type will always give the same lock.
typedef
hpx::util::static_<spinlock,
                   global_device_list_tag>  static_device_list_lock_type;

// The shutdown hook for clearing the device list on hpx::finalize()
static void clear_device_list()
{

    // Lock the list
    static_device_list_lock_type device_lock;
    boost::lock_guard<spinlock> lock(device_lock.get());

    // get static device list
    static_device_list_type devices;

    // clear the devices list
    devices.get().clear();

}


///////////////////////////////////////////////////
/// HPX Registration Stuff
///
HPX_REGISTER_PLAIN_ACTION(hpx::opencl::server::get_devices_action,
                          opencl_get_devices_action);

///////////////////////////////////////////////////
/// Local functions
///
static std::vector<int> 
parse_version_string(std::string version_str)
{

    try{
       
        // Make sure the version string starts with "OpenCL "
        BOOST_ASSERT(version_str.compare(0, 7, "OpenCL ") == 0);

        // Cut away the "OpenCL " in front of the version string
        version_str = version_str.substr(7);
    
        // Cut away everything behind the version number
        version_str = version_str.substr(0, version_str.find(" "));
        
        // Get major version string
        std::string version_str_major = 
                           version_str.substr(0, version_str.find("."));

        // Get minor version string
        std::string version_str_minor = 
                           version_str.substr(version_str_major.size() + 1);

        // create output vector
        std::vector<int> version_numbers(2);

        // Parse version number
        version_numbers[0] = ::atoi(version_str_major.c_str());
        version_numbers[1] = ::atoi(version_str_minor.c_str());

        // Return the parsed version number
        return version_numbers;

    } catch (const std::exception & ex) {
        hpx::cerr << "Error while parsing OpenCL Version!" << hpx::endl;
        std::vector<int> version_numbers(2);
        version_numbers[0] = -1;
        version_numbers[1] = 0;
        return version_numbers;
    }

}


///////////////////////////////////////////////////
/// Implementations
///

// This method initializes the devices-list if it's not done yet.
void
ensure_device_components_initialization()
{

    // Lock the list
    static_device_list_lock_type device_lock;
    boost::lock_guard<spinlock> lock(device_lock.get());

    // get static device list
    static_device_list_type devices;

    // Register the shutdown hook to empty the device list before shutdown
    if(device_shutdown_hook_initialized == false)
    {
        hpx::get_runtime_ptr()->add_pre_shutdown_function(&clear_device_list);
        device_shutdown_hook_initialized = true;
    }

    // Don't initialize if already initialized
    if(device_list_initialized != false)
        return;

    // Reset the device list
    devices.get().clear();

    // Declairing the cl error code variable
    cl_int err;

    // Query for number of available platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    cl_ensure(err, "clGetPlatformIDs()");

    // Retrieve platforms
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), NULL);
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
                                  devices_on_platform.data(), NULL);
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

    device_list_initialized = true;

}
    

std::vector<hpx::opencl::device>
hpx::opencl::server::get_devices(cl_device_type type,
                                 std::string min_cl_version)
{

    // Parse required OpenCL version
    std::vector<int> required_version = parse_version_string(min_cl_version);

    // Create the list of device clients
    ensure_device_components_initialization();

    // Lock the list
    static_device_list_lock_type device_lock;
    boost::lock_guard<spinlock> lock(device_lock.get());

    // get static device list
    static_device_list_type devices;

    // Generate a list of suitable devices
    std::vector<hpx::opencl::device> suitable_devices;
    BOOST_FOREACH( const std::vector<hpx::opencl::device>::value_type& device,
                   devices.get())
    {
        // Get device OpenCL version
        std::vector<char> cl_version_string_vec =
                                device.get_device_info(CL_DEVICE_VERSION).get();

        // Make String out of char array
        std::string cl_version_string (cl_version_string_vec.begin(),
                                       cl_version_string_vec.end());
    
        // Parse OpenCL version
        std::vector<int> device_cl_version = 
                                        parse_version_string(cl_version_string);

        // Check if device supports required version
        if(device_cl_version[0] < required_version[0]) continue;
        if(device_cl_version[0] == required_version[0])
        {
            if(device_cl_version[1] < required_version[1]) continue;
        }

        // Check for requested device type
        std::vector<char> device_type_string = 
                                   device.get_device_info(CL_DEVICE_TYPE).get();
        cl_device_type device_type = *((cl_device_type*)
                                                   (device_type_string.data()));
        if(!(device_type & type)) continue;

        // TODO filter devices
        suitable_devices.push_back(device);
    }

    // Return the devices found
    return suitable_devices;

}



