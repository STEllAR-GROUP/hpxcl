#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>

#include "../../opencl/std.hpp"
#include "../../opencl/device.hpp"

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

    // Get list of available OpenCL Devices
    std::vector<clx_device_id> devices = clGetDeviceIDs(hpx::find_here(),
                                                        CL_DEVICE_TYPE_ALL);

    // Check whether there are any devices
    if(devices.size() < 1)
    {
        hpx::cerr << "No OpenCL devices found!" << hpx::endl;
        return hpx::finalize();
    }

    // Create a device component from the first device found
    device cldevice(
             hpx::components::new_<server::device>(hpx::find_here(), devices[0])
                       );

    // Create a buffer where the device can write to
    buffer outbuffer = cldevice.clCreateBuffer(CL_MEM_WRITE_ONLY, 13);

    // Create the hello_world device program
    program prog = cldevice.clCreateProgramWithSource(hello_world_src);

    // Compile the program
    prog.build();

    // Create hello_world kernel
    kernel hello_world_kernel = prog.create_kernel("hello_world");

    // Set our buffer as argument
    hello_world_kernel.set_arg(0, outbuffer);

    // Run the kernel
    size_t offset = 0;
    size_t size = 13;
    event kernel_event = hello_world_kernel.enqueue(1, &offset, &size,
                                                          (size_t*) NULL).get(); 

    // Read the buffer
    event read_event = outbuffer.clEnqueueReadBuffer(0, 13, kernel_event).get();

    // Retrieve the read data
    boost::shared_ptr<std::vector<char>> data_ptr = read_event.get_data().get();

    // Write the data to hpx::cout
    hpx::cout << &(*data_ptr)[0] << hpx::endl;

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


