// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>

#include "../../opencl.hpp"

using namespace hpx::opencl;

static const char hello_world_src[] = 
"                                                                          \n"
"   __kernel void hello_world(__global char * out)                         \n"
"   {                                                                      \n"
"       char in [] = \"Hello World!\";                                     \n"
"       size_t tid = get_global_id(0);                                     \n"
"       out[tid] = in[tid];                                                \n"
"   }                                                                      \n"
"                                                                          \n";


// hpx_main, is the actual main called by hpx
int hpx_main(int argc, char* argv[])
{
    {
        
        // Get list of available OpenCL Devices.
        std::vector<device> devices = get_devices( hpx::find_here(),
                                                   CL_DEVICE_TYPE_ALL,
                                                   1.1f ).get();
    
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
        hpx::lcos::shared_future<event> kernel_event =
                                                hello_world_kernel.enqueue(dim); 

        // Start reading the buffer (With kernel_event dependency.
        //                           All hpxcl enqueue calls are nonblocking.)
        event read_event = outbuffer.enqueue_read(0, 13, kernel_event).get();
    
        // Get the data (blocks until read_event finishes)
        boost::shared_ptr<std::vector<char>> data_ptr = read_event.get_data().get();
    
        // Write the data to hpx::cout
        hpx::cout << data_ptr->data() << hpx::endl;
        
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


