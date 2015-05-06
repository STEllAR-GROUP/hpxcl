// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>

#include "../../opencl.hpp"
#include <sstream>

using namespace hpx::opencl;

static void printinfo(size_t i, size_t j, std::string info_type,
                      std::string info)
{

    hpx::cout << " " << i << "." << j << ". " 
              << info_type << ": " << info << hpx::endl;

}
            
static std::string
device_uint_to_string(hpx::opencl::info && info_data)
{

    cl_uint res = static_cast<cl_uint>(info_data);
    
    std::stringstream ss;
    ss << res;
    return ss.str();

}
static std::string
device_type_to_string(hpx::opencl::info && info_data)
{

    cl_device_type type = static_cast<cl_device_type>(info_data);

    std::vector<std::string> typelist;

    if(type & CL_DEVICE_TYPE_CPU)
        typelist.push_back("cpu");

    if(type & CL_DEVICE_TYPE_GPU)
        typelist.push_back("gpu");

    if(type & CL_DEVICE_TYPE_ACCELERATOR)
        typelist.push_back("accelerator");

    if(type & CL_DEVICE_TYPE_DEFAULT)
        typelist.push_back("default");

#ifdef CL_VERSION_1_2
    if(type & CL_DEVICE_TYPE_CUSTOM)
        typelist.push_back("custom");
#endif

    std::string result = "";

    for(size_t i = 0 ; i < typelist.size(); i++)
    {

        if(i > 0)
            result += ", ";

        result += typelist[i];

    }

    return result;

}

// hpx_main, is the actual main called by hpx
int hpx_main(int argc, char* argv[])
{
    {
        
        // Get list of available OpenCL Devices.
        std::vector<device> devices = get_all_devices(CL_DEVICE_TYPE_ALL,
                                                   "OpenCL 1.0" ).get();
    
        // Check whether there are any devices
        if(devices.size() < 1)
        {
            hpx::cerr << "No OpenCL devices found!" << hpx::endl;
            return hpx::finalize();
        }
    
        hpx::cout << hpx::endl << "Devices:" << hpx::endl << hpx::endl;

        // print a lot of information about every device
        size_t i = 1;
        for(auto &cldevice : devices)
        {
   
            size_t j = 1;

            // generate string
            std::string str;
             
            // print name
            str = static_cast<std::string>(
                        cldevice.get_device_info(CL_DEVICE_NAME));
            hpx::cout << i << ". " << str << hpx::endl;

            // print platform name
            str = static_cast<std::string>(
                        cldevice.get_platform_info(CL_PLATFORM_NAME));
            printinfo(i, j++, "Platform", str);

            // print supported opencl version
            str = static_cast<std::string>(
                        cldevice.get_device_info(CL_DEVICE_VERSION));
            printinfo(i, j++, "OpenCL Version", str);

            // print device type
            str = device_type_to_string(
                        cldevice.get_device_info(CL_DEVICE_TYPE));
            printinfo(i, j++, "Type", str);

            // print driver version
            str = static_cast<std::string>(
                        cldevice.get_device_info(CL_DRIVER_VERSION));
            printinfo(i, j++, "Driver Version", str);

            // print vendor
            str = device_uint_to_string(
                        cldevice.get_device_info(CL_DEVICE_VENDOR_ID));
            str += " - ";
            str += static_cast<std::string>(
                        cldevice.get_device_info(CL_DEVICE_VENDOR));
            printinfo(i, j++, "Vendor", str);

            // print profile
            str = static_cast<std::string>(
                        cldevice.get_device_info(CL_DEVICE_PROFILE));
            printinfo(i, j++, "Profile", str);

            // print compiler c version
            str = static_cast<std::string>(
                        cldevice.get_device_info(CL_DEVICE_OPENCL_C_VERSION));
            printinfo(i, j++, "Compiler Version", str);
            
            /*** TO BE CONTINUED ***/


            // add newline before starting a new device
            hpx::cout << hpx::endl;

            i++;
        }
    
        

    }
    
    // End the program
    return hpx::finalize();
}

// Main, initializes HPX
int main(int argc, char* argv[]){

    // initialize HPX, run hpx_main
    hpx::start(argc, argv);

    // wait for hpx::finalize being called
    return hpx::stop();
}


