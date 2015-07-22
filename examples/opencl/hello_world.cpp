// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>

#include "../../opencl.hpp"

using namespace hpx::opencl;

static const char hello_world_src_str[] = 
"                                                                          \n"
"   __kernel void hello_world(__global char * out)                         \n"
"   {                                                                      \n"
"       char in [] = \"Hello World!\";                                     \n"
"       size_t tid = get_global_id(0);                                     \n"
"       out[tid] = in[tid];                                                \n"
"   }                                                                      \n"
"                                                                          \n";

typedef hpx::serialization::serialize_buffer<char> buffer_type;

static buffer_type hello_world_src( hello_world_src_str,
                                    sizeof(hello_world_src_str),
                                    buffer_type::init_mode::reference );
                    

// hpx_main, is the actual main called by hpx
int main(int argc, char* argv[])
{

    // Get list of available OpenCL Devices.
    std::vector<device> devices = get_all_devices(CL_DEVICE_TYPE_ALL,
                                                  "OpenCL 1.1" ).get();

    // Check whether there are any devices
    if(devices.size() < 1)
    {
        hpx::cerr << "No OpenCL devices found!" << hpx::endl;
        return hpx::finalize();
    }

    // Create a device component from the first device found
    device cldevice = devices[0];

    // Create a buffer
    buffer outbuffer = cldevice.create_buffer(CL_MEM_WRITE_ONLY, 13);

    // Create the hello_world device program
    program prog = cldevice.create_program_with_source(hello_world_src);

    // Compile the program
    prog.build();

    // Create hello_world kernel
    kernel hello_world_kernel = prog.create_kernel("hello_world");

    // Set our buffer as argument
    hello_world_kernel.set_arg(0, outbuffer);

    // Run the kernel
    hpx::opencl::work_size<1> dim;
    dim[0].offset = 0;
    dim[0].size = 13;
    hpx::future<void> kernel_future = hello_world_kernel.enqueue(dim); 

    // Start reading the buffer ( With kernel_future as dependency.
    //                            All hpxcl enqueue calls are nonblocking. )
    auto read_future = outbuffer.enqueue_read(0, 13, kernel_future);

    // Wait for the data to arrive
    auto data = read_future.get();

    // Write the data to hpx::cout
    hpx::cout << data.data() << hpx::endl;

    return 0;
}

