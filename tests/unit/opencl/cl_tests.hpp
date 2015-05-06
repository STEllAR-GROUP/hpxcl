// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
//#include <hpx/util/static.hpp>
#include <hpx/include/iostreams.hpp>

#include "../../../opencl.hpp"

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

// the main test function
static void cl_test(hpx::opencl::device);

/*
#define TEST_CL_BUFFER(buffer, value)                                          \
{                                                                              \
    boost::shared_ptr<std::vector<char>> out1 =                                \
                       buffer.enqueue_read(0, DATASIZE).get().get_data().get();\
    HPX_TEST_EQ(std::string((const char*)(value)), std::string(out1->data()));  \
}                                                           
*/

static hpx::opencl::device init(variables_map & vm)
{

    std::size_t device_id = 0;

    if (vm.count("deviceid"))
        device_id = vm["deviceid"].as<std::size_t>();

    // Query devices
    std::vector<hpx::opencl::device> devices
            = hpx::opencl::get_devices( hpx::find_here(),
                                        CL_DEVICE_TYPE_ALL, "OpenCL 1.1").get();
    HPX_TEST(devices.size() >= device_id);

    // Choose device
    hpx::opencl::device cldevice = devices[device_id];

    // Test whether get_device_info works
    std::string version = cldevice.get_device_info<CL_DEVICE_VERSION>();

    // Test whether version is a valid OpenCL version string
    std::string versionstring = std::string("OpenCL ");
    HPX_TEST(0 == version.compare(0, versionstring.length(), versionstring));

    // Write Info Code
    hpx::cout << "Device ID:  " << device_id << " / " << devices.size()
                                << hpx::endl;
    hpx::cout << "Version:    " << version << hpx::endl;
    hpx::cout << "Name:       " << cldevice.get_device_info<CL_DEVICE_NAME>()
                                << hpx::endl;
    hpx::cout << "Vendor:     " << cldevice.get_device_info<CL_DEVICE_VENDOR>()
                                << hpx::endl;
    hpx::cout << "Profile:    " << cldevice.get_device_info<CL_DEVICE_PROFILE>()
                                << hpx::endl;

    // Test for valid device client
    HPX_TEST(cldevice.get_gid());

    // return the device
    return cldevice;

}

int hpx_main(variables_map & vm)
{
    {
        hpx::opencl::device cldevice = init(vm);   
        cl_test(cldevice);
    }
    
    hpx::finalize();
    return hpx::util::report_errors();
}



///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");
    cmdline.add_options()
        ( "deviceid"
        , value<std::size_t>()->default_value(0)
        , "the ID of the device we will run our tests on") ;

    return hpx::init(cmdline, argc, argv);
}
