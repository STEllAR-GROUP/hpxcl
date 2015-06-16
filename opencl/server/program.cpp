// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The Header of this class
#include "program.hpp"

// HPXCL tools
#include "../tools.hpp"

// other hpxcl dependencies
#include "device.hpp"
#include "util/hpx_cl_interop.hpp"
#include "kernel.hpp"

// HPX dependencies
#include <hpx/include/thread_executors.hpp>

using hpx::opencl::server::program;


// Constructor
program::program()
{}

// External destructor.
// This is needed because OpenCL calls only run properly on large stack size.
static void program_cleanup(uintptr_t program_id_ptr)
{

    cl_int err;

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    cl_program program_id = reinterpret_cast<cl_program>(program_id_ptr);

    // Release the device memory
    if(program_id)
    {
        err = clReleaseProgram(program_id);
        cl_ensure_nothrow(err, "clReleaseProgram()");
    }
}

// Destructor
program::~program()
{

    hpx::threads::executors::default_executor exec(
                                          hpx::threads::thread_priority_normal,
                                          hpx::threads::thread_stacksize_large);

    // run dectructor in a thread, as we need it to run on a large stack size
    hpx::async( exec, &program_cleanup, reinterpret_cast<uintptr_t>(program_id))
                                                                        .wait();


}


void
program::init_with_source( hpx::naming::id_type device_id, 
                           hpx::serialization::serialize_buffer<char> src )
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    this->parent_device_id = std::move(device_id);
    this->parent_device = hpx::get_ptr
                          <hpx::opencl::server::device>(parent_device_id).get();
    this->program_id = NULL;

    // Retrieve the context from parent class
    cl_context context = parent_device->get_context();

    // The opencl error variable
    cl_int err;

    // Set up data for OpenCL call
    HPX_ASSERT(src.size() > 0);
    std::size_t src_size = src.size();
    const char* src_data = src.data();
    if(src_data[src_size - 1] == '\0'){
        // Decrease one if zero-terminated, as
        // OpenCL specifies 'length of source string excluding null terminator'
        src_size --;
    }

    // Create the cl_program
    program_id = clCreateProgramWithSource( context, 1, &src_data, &src_size,
                                            &err );
    cl_ensure(err, "clCreateProgramWithSource()");

}

std::string
program::acquire_build_log()
{
    cl_int err;

    std::size_t build_log_size;

    // Query size
    err = clGetProgramBuildInfo(program_id, parent_device->get_device_id(),
                                CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
   
    // Create buffer
    std::vector<char> buf(build_log_size);

    // Get log
    err = clGetProgramBuildInfo(program_id, parent_device->get_device_id(),
                                CL_PROGRAM_BUILD_LOG, build_log_size,
                                buf.data(), NULL);

    // make build log look nice in exception
    std::stringstream sstream;
    sstream << std::endl << std::endl;
    sstream << "//////////////////////////////////////" << std::endl;
    sstream << "/// OPENCL BUILD LOG" << std::endl;
    sstream << "///" << std::endl;
    sstream << std::endl << buf.data() << std::endl;
    sstream << "///" << std::endl;
    sstream << "/// OPENCL BUILD LOG END" << std::endl;
    sstream << "//////////////////////////////////////" << std::endl;
    sstream << std::endl;
    
    // return the nice looking error string.
    return sstream.str();

}

void
program::throw_on_build_errors(const char* function_name){

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    cl_int err;
    cl_build_status build_status;

    // Read build status
    err = clGetProgramBuildInfo( program_id, parent_device->get_device_id(),
                                 CL_PROGRAM_BUILD_STATUS,
                                 sizeof(cl_build_status), &build_status, NULL );
    cl_ensure(err, "clGetProgramBuildInfo()");

    // Throw if build did not succeed
    if(build_status != CL_BUILD_SUCCESS)
    {
        HPX_THROW_EXCEPTION(hpx::no_success, function_name,
                            std::string("A build error occured!") +
                            acquire_build_log());
    }
}

struct build_callback_args{
    hpx::runtime* rt;
    hpx::lcos::local::promise<void>* promise;
};

static void CL_CALLBACK
build_callback( cl_program program_id, void* user_data )
{
    // Cast arguments
    build_callback_args* args =
        static_cast<build_callback_args*>(user_data);

    // Send exec status to waiting future
    using hpx::opencl::server::util::set_promise_from_external;
    set_promise_from_external ( args->rt, args->promise );
}

void
program::build(std::string options)
{
    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    cl_int err;

    // fetch device id from parent device
    cl_device_id device_id = parent_device->get_device_id();

    // Create a new promise
    hpx::lcos::local::promise<void> promise;

    // Retrieve the future
    hpx::future<void> future = promise.get_future(); 

    // Create args for build_callback
    build_callback_args args;
    args.rt = hpx::get_runtime_ptr();
    args.promise = &promise;

    // Initialize compilation
    err = clBuildProgram( program_id, 1, &device_id, options.c_str(),
                          &build_callback, &args );

    // ignore CL_BUILD_PROGRAM_FAILURE.
    // we handle this case in throw_on_build_errors()
    if(err != CL_BUILD_PROGRAM_FAILURE)
        cl_ensure(err, "clBuildProgram()");

    // Wait for compilation to finish
    future.wait();

    // check build status 
    throw_on_build_errors("clBuildProgram()");

}

hpx::serialization::serialize_buffer<char>
program::get_binary()
{
    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    typedef hpx::serialization::serialize_buffer<char> buffer_type;
    cl_int err;

    // get number of devices
    cl_uint num_devices;
    err = clGetProgramInfo(program_id, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint),
                           &num_devices, NULL);
    cl_ensure(err, "clGetProgramInfo()");
    
    // ensure that only one device is associated
    if(num_devices != 1)
    {
        HPX_THROW_EXCEPTION(hpx::internal_server_error, "program::get_binary()",
                            "Internal Error: More than one device linked!");
    }

    // get binary size
    std::size_t binary_size;
    err = clGetProgramInfo(program_id, CL_PROGRAM_BINARY_SIZES,
                           sizeof(std::size_t), &binary_size, NULL);
    cl_ensure(err, "clGetProgramInfo()");

    // ensure that there actually is binary code
    if(binary_size == 0)
    {
        HPX_THROW_EXCEPTION(hpx::no_success, "program::get_binary()",
                            "Unable to fetch binary code!");
    }

    // get binary code
    buffer_type binary( new char[binary_size], binary_size,
                        buffer_type::init_mode::take );
    char* binary_ptr = binary.data();
    err = clGetProgramInfo( program_id, CL_PROGRAM_BINARIES,
                            sizeof(unsigned char*),
                            &binary_ptr,
                            NULL );
    cl_ensure(err, "clGetProgramInfo()");

    // return vector
    return binary;

}

hpx::naming::id_type
program::create_kernel(std::string kernel_name)
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    // Create new kernel
    hpx::id_type kernel = hpx::components::new_<hpx::opencl::server::kernel>
                                                     ( hpx::find_here() ).get();

    // Initialize kernel locally
    auto kernel_server = hpx::get_ptr<hpx::opencl::server::kernel>(kernel).get();

    kernel_server->init(parent_device_id, program_id, kernel_name);

    return kernel;

}
