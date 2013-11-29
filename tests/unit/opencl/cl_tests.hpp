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


hpx::opencl::device cldevice;

hpx::naming::id_type here;

void init(variables_map & vm)
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

    // Test whether get_device_info works
    std::vector<char> version_char_array = 
            hpx::opencl::get_device_info(here, devices[device_id],
                                         CL_DEVICE_VERSION).get();

    // Convert char array to string
    std::string version(version_char_array.begin(), version_char_array.end());

    // Test whether version is a valid OpenCL version string
    std::string versionstring = std::string("OpenCL ");
    HPX_TEST(0 == version.compare(0, versionstring.length(), versionstring));

    // Create a device
    typedef hpx::opencl::server::device device_type;
    hpx::opencl::device new_cldevice(
                        hpx::components::new_<device_type>(here,
                                                           devices[device_id]));
    cldevice = new_cldevice;

}


