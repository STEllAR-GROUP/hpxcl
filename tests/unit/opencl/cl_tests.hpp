// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "../../../opencl/device.hpp"

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;


static boost::shared_ptr<hpx::opencl::device> cldevice;
static hpx::opencl::clx_device_id clx_device;
static hpx::naming::id_type here;

static void cl_test();

static std::string get_cl_info(cl_device_info info_type)
{

    return hpx::opencl::get_device_info_string(here, clx_device, info_type).get();

}

static void init(variables_map & vm)
{

    here = hpx::find_here();

    std::size_t device_id = 0;

    if (vm.count("deviceid"))
        device_id = vm["deviceid"].as<std::size_t>();

    // Query devices
    std::vector<hpx::opencl::clx_device_id> devices
            = hpx::opencl::get_device_ids(here,
                CL_DEVICE_TYPE_ALL, 1.1f).get();
    HPX_TEST(devices.size() >= device_id);

    // Choose device
    clx_device = devices[device_id];

    // Test whether get_device_info works
    std::string version = get_cl_info(CL_DEVICE_VERSION);

    // Test whether version is a valid OpenCL version string
    std::string versionstring = std::string("OpenCL ");
    HPX_TEST(0 == version.compare(0, versionstring.length(), versionstring));

    // Write Info Code
    hpx::cout << "Device ID:  " << device_id << " / " << devices.size() << hpx::endl;
    hpx::cout << "Version:    " << version << hpx::endl;
    hpx::cout << "Name:       " << get_cl_info(CL_DEVICE_NAME) << hpx::endl;
    hpx::cout << "Vendor:     " << get_cl_info(CL_DEVICE_VENDOR) << hpx::endl;
    hpx::cout << "Profile:    " << get_cl_info(CL_DEVICE_PROFILE) << hpx::endl;

    // Create a device
    typedef hpx::opencl::server::device device_type;
    cldevice = boost::make_shared<hpx::opencl::device>(
                    hpx::components::new_<device_type>(here, devices[device_id])
                                                                    );

    HPX_TEST(cldevice->get_gid());

}

static void shutdown()
{

    cldevice.reset();
    
}

int hpx_main(variables_map & vm)
{
    {
        init(vm);   
        cl_test();
        shutdown();
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
