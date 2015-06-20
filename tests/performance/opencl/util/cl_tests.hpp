// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <hpx/config.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
//#include <hpx/util/static.hpp>
#include <hpx/include/iostreams.hpp>

#include "../../../../opencl.hpp"

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

// the main test function
static void cl_test(hpx::opencl::device, hpx::opencl::device);

#define die( message )                                                          \
{                                                                               \
    HPX_THROW_EXCEPTION(hpx::no_success, "die()", (message));                   \
}

#define CREATE_BUFFER(name, data)                                               \
    static const buffer_type name(data, sizeof(data),                           \
                                  buffer_type::init_mode::reference)

#define COMPARE_RESULT_INT( result_data, correct_result )                       \
{                                                                               \
    auto lhs = result_data;                                                     \
    auto rhs = correct_result;                                                  \
    if(lhs.size() != rhs.size()){                                               \
        die("Result is incorrect! (Sizes don't match)");                        \
    }                                                                           \
    for(std::size_t i = 0; i < lhs.size(); i++){                                \
        std::cout << std::hex << lhs[i] << "-" << rhs[i] << std::endl;          \
        if(lhs[i] != rhs[i]){                                                   \
            die("Result is incorrect!");                                        \
    }                                                                           \
}

typedef hpx::serialization::serialize_buffer<char> buffer_type;
typedef hpx::serialization::serialize_buffer<uint32_t> intbuffer_type;

std::string to_string(buffer_type buf){
    std::size_t length = 0; 
    while(length < buf.size())
    {
        if(buf[length] == '\0') break;
        length++;
    }
    return std::string(buf.data(), buf.data() + length);
}

#define COMPARE_RESULT( result_data, correct_result )                           \
{                                                                               \
    auto lhs = result_data;                                                     \
    auto rhs = correct_result;                                                  \
    if(lhs.size() != rhs.size()){                                               \
        die("Result is incorrect! (Sizes don't match)");                        \
    }                                                                           \
    std::string correct_string = to_string(rhs);                                \
    std::string result_string = to_string(lhs);                                 \
    if(correct_string != result_string){                                        \
        die("Result is incorrect!");                                            \
    }                                                                           \
}

static void print_testdevice_info(hpx::opencl::device & cldevice,
                                  std::size_t device_id,
                                  std::size_t num_devices){

    // Test whether get_device_info works
    std::string version = cldevice.get_device_info<CL_DEVICE_VERSION>().get();

    // Write Info Code
    std::cout << "Device ID:  " << device_id << " / " << num_devices
                                << std::endl;
    std::cout << "Device GID: " << cldevice.get_gid() << std::endl;
    std::cout << "Version:    " << version << std::endl;
    std::cout << "Name:       " << cldevice.get_device_info<CL_DEVICE_NAME>().get()
                                << std::endl;
    std::cout << "Vendor:     " << cldevice.get_device_info<CL_DEVICE_VENDOR>().get()
                                << std::endl;
    std::cout << "Profile:    " << cldevice.get_device_info<CL_DEVICE_PROFILE>().get()
                                << std::endl;
}

static std::vector<hpx::opencl::device> init(variables_map & vm)
{

    std::size_t device_id = 0;

    if (vm.count("deviceid"))
        device_id = vm["deviceid"].as<std::size_t>();

    // Try to get remote devices
    std::vector<hpx::opencl::device> remote_devices
            = hpx::opencl::get_local_devices( CL_DEVICE_TYPE_ALL,
                                               "OpenCL 1.1" ).get();
    std::vector<hpx::opencl::device> local_devices
            = hpx::opencl::get_local_devices( CL_DEVICE_TYPE_ALL,
                                              "OpenCL 1.1" ).get();
    if(remote_devices.empty()) die("No remote devices found!");
    if(local_devices.empty()) die("No local devices found!");
    if(local_devices.size() <= device_id || remote_devices.size() <= device_id)
        die("deviceid is out of range!");

    // Choose device
    hpx::opencl::device local_device  = local_devices[device_id];
    hpx::opencl::device remote_device = remote_devices[device_id];

    // Print info
    std::cout << "Local device:" << std::endl;
    print_testdevice_info(local_device, device_id, local_devices.size());
    std::cout << "Remote device:" << std::endl;
    print_testdevice_info(remote_device, device_id, remote_devices.size());

    // return the devices
    std::vector<hpx::opencl::device> devices;
    devices.push_back(local_device);
    devices.push_back(remote_device);
    return devices;

}

int hpx_main(variables_map & vm)
{
    {
        auto devices = init(vm);   
        std::cout << std::endl;
        cl_test(devices[0], devices[1]);
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
