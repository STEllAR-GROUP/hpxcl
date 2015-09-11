// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>

#include "../../cuda.hpp"

#define DEBUG 

using namespace hpx::cuda;

static const char kernel_src[] =
"                                                                           "
"  __global__ void sum(int* array , int n, int* count){ 				  \n"
" for (int i = blockDim.x * blockIdx.x + threadIdx.x;					  \n"
"         i < n;														  \n"
"         i += gridDim.x * blockDim.x)									  \n"
"    {																	  \n"
"        atomicAdd(&(count[0]), array[i]);								  \n"
"    }																	  \n"
"}                                             							  \n";

//typedef hpx::serialization::serialize_buffer<char> buffer_type;

//static buffer_type kernel_src( kernel_str,
  //                                  sizeof(kernel_str),
    //                                buffer_type::init_mode::reference );


// hpx_main, is the actual main called by hpx
int main(int argc, char* argv[])
{

    // Get list of available OpenCL Devices.
    std::vector<device> devices = get_all_devices(2,0).get();

    // Check whether there are any devices
    if(devices.size() < 1)
    {
        hpx::cerr << "No CUDA devices found!" << hpx::endl;
        return hpx::finalize();
    }

    // Create a device component from the first device found
    device cudaDevice = devices[0];

    // Create a buffer
    //buffer outbuffer = cldevice.create_buffer(CL_MEM_WRITE_ONLY, 13);

    // Create the hello_world device program
    program prog = cudaDevice.create_program_with_source(kernel_src).get();

    // Add compiler flags for compiling the kernel

    std::vector<std::string> flags;
    std::string mode = "--gpu-architecture=compute_";
    mode.append(std::to_string(cudaDevice.get_device_architecture_major().get()));

    mode.append(std::to_string(cudaDevice.get_device_architecture_minor().get()));

    flags.push_back(mode);

    // Compile the program

#ifdef DEBUG
    prog.build(flags,1);
#else
    prog.build(flags);
#endif
    // Create hello_world kernel
    //kernel hello_world_kernel = prog.create_kernel("hello_world");

    // Set our buffer as argument
    //hello_world_kernel.set_arg(0, outbuffer);

    // Run the kernel
    //hpx::opencl::work_size<1> dim;
   // dim[0].offset = 0;
   // dim[0].size = 13;
  //  hpx::future<void> kernel_future = hello_world_kernel.enqueue(dim);

    // Start reading the buffer ( With kernel_future as dependency.
    //                            All hpxcl enqueue calls are nonblocking. )
  //  auto read_future = outbuffer.enqueue_read(0, 13, kernel_future);

    // Wait for the data to arrive
 //   auto data = read_future.get();

    // Write the data to hpx::cout
 //   hpx::cout << data.data() << hpx::endl;

    return EXIT_SUCCESS;
}





