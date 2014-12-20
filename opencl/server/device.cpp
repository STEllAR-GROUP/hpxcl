// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Default includes
#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

// The Header of this class
#include "device.hpp"

// HPXCL tools
#include "../tools.hpp"

// HPX dependencies
#include <hpx/include/thread_executors.hpp>

// OpenCL
#include <CL/cl.h>

// other hpxcl dependencies
#include "buffer.hpp"

using namespace hpx::opencl::server;


// Constructor
device::device()
{}

// External destructor.
// This is needed because OpenCL calls only run properly on large stack size.
static void device_cleanup(uintptr_t command_queue_ptr,
                           uintptr_t context_ptr)
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    cl_int err;

    cl_command_queue command_queue = reinterpret_cast<cl_command_queue>(command_queue_ptr);
    cl_context context = reinterpret_cast<cl_context>(context_ptr);

    // Release command queue
    if(command_queue)
    {
        err = clFinish(command_queue);
        cl_ensure_nothrow(err, "clFinish()");
        err = clReleaseCommandQueue(command_queue);
        cl_ensure_nothrow(err, "clReleaseCommandQueue()");
        command_queue = NULL; 
    }
    
    // Release context
    if(context)
    {
        err = clReleaseContext(context);
        cl_ensure_nothrow(err, "clReleaseContext()");
        context = NULL;
    }

}

// Destructor
device::~device()
{

    hpx::threads::executors::default_executor exec(
                                          hpx::threads::thread_priority_normal,
                                          hpx::threads::thread_stacksize_large);

    // run dectructor in a thread, as we need it to run on a large stack size
    hpx::async( exec, &device_cleanup, (uintptr_t)command_queue,
                                       (uintptr_t)context).wait();

}

// Initialization function.
// Needed because cl_device_id can not be serialized.
void
device::init(cl_device_id _device_id, bool enable_profiling)
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    this->device_id = _device_id;

    cl_int err;
    
    // Retrieve platformID
    err = clGetDeviceInfo(this->device_id, CL_DEVICE_PLATFORM,
                          sizeof(platform_id), &platform_id, NULL);
    cl_ensure(err, "clGetDeviceInfo()");

    // Create Context
    cl_context_properties context_properties[] = 
                        {CL_CONTEXT_PLATFORM,
                         (cl_context_properties) platform_id,
                         0};
    context = clCreateContext(context_properties,
                              1,
                              &this->device_id,
                              error_callback,
                              this,
                              &err);
    cl_ensure(err, "clCreateContext()");

    // Get supported device queue properties
    hpx::util::serialize_buffer<char> supported_queue_properties_data = 
                                    get_device_info(CL_DEVICE_QUEUE_PROPERTIES);
    cl_command_queue_properties supported_queue_properties =
                *( reinterpret_cast<cl_command_queue_properties *>(
                                       supported_queue_properties_data.data()));

    // Initialize command queue properties
    cl_command_queue_properties command_queue_properties = 0;

    // If supported, add OUT_OF_ORDER_EXEC_MODE
    if(supported_queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
        command_queue_properties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;

    // If supported and wanted, add PROFILING
    if(enable_profiling &&
                       (supported_queue_properties & CL_QUEUE_PROFILING_ENABLE))
        command_queue_properties |= CL_QUEUE_PROFILING_ENABLE;

    // Create Command Queue
    command_queue = clCreateCommandQueue(context, device_id,
                                         command_queue_properties, &err);
    cl_ensure(err, "clCreateCommandQueue()");
}


hpx::util::serialize_buffer<char>
device::get_device_info(cl_device_info info_type)
{
    
    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    // Declairing the cl error code variable
    cl_int err;

    // Query for size
    std::size_t param_size;
    err = clGetDeviceInfo(device_id, info_type, 0, NULL, &param_size);
    cl_ensure(err, "clGetDeviceInfo()");

    // Retrieve
    hpx::util::serialize_buffer<char> info( new char[param_size],
                                            param_size,
                                            hpx::util::serialize_buffer<char>::take);
    err = clGetDeviceInfo(device_id, info_type, param_size, info.data(), 0);
    cl_ensure(err, "clGetDeviceInfo()");

    // Return
    return info;

}


hpx::util::serialize_buffer<char>
device::get_platform_info(cl_platform_info info_type)
{
    
    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    // Declairing the cl error code variable
    cl_int err;

    // Query for size
    std::size_t param_size;
    err = clGetPlatformInfo(platform_id, info_type, 0, NULL, &param_size);
    cl_ensure(err, "clGetPlatformInfo()");

    // Retrieve
    hpx::util::serialize_buffer<char> info( new char[param_size],
                                            param_size,
                                            hpx::util::serialize_buffer<char>::take);
    err = clGetPlatformInfo(platform_id, info_type, param_size, info.data(), 0);
    cl_ensure(err, "clGetPlatformInfo()");

    // Return
    return info;

}


void CL_CALLBACK
device::error_callback(const char* errinfo, const void* info, std::size_t info_size,
                                                void* _thisp)
{
    device* thisp = (device*) _thisp;
    hpx::cerr << "device(" << thisp->device_id << "): CONTEXT_ERROR: "
             << errinfo << hpx::endl;
}

cl_context
device::get_context()
{
    return context;
}

hpx::id_type
device::create_buffer( cl_mem_flags flags, std::size_t size )
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    // Create new buffer
    hpx::id_type buf = hpx::components::new_<hpx::opencl::server::buffer>
                                                     ( hpx::find_here() ).get();

    // Initialzie buffer locally
    boost::shared_ptr<hpx::opencl::server::buffer> buffer_server = 
                        hpx::get_ptr<hpx::opencl::server::buffer>( buf ).get();

    buffer_server->init(get_gid(), flags, size);

    return buf;           
}
