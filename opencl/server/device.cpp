// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "device.hpp"

#include "../tools.hpp"

#include <CL/cl.h>

using namespace hpx::opencl::server;


// Constructor
device::device()
{}

// External destructor action.
// This is needed because OpenCL calls only run properly on large stack size.
namespace hpx { namespace opencl { namespace server {
    static void device_cleanup(uintptr_t command_queue_ptr,
                               uintptr_t context_ptr)
    {
        cl_int err;

        cl_command_queue command_queue = (cl_command_queue) command_queue_ptr;
        cl_context context = (cl_context) context_ptr;
    
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
    HPX_DEFINE_PLAIN_ACTION(device_cleanup, device_cleanup_action);
}}}

HPX_ACTION_USES_LARGE_STACK(hpx::opencl::server::device_cleanup_action);
HPX_REGISTER_PLAIN_ACTION_DECLARATION(hpx::opencl::server::device_cleanup_action);
HPX_REGISTER_PLAIN_ACTION(hpx::opencl::server::device_cleanup_action,
                          hpx_opencl_server_device_cleanup_action);

// Destructor
device::~device()
{

    // run dectructor in a thread, as we need it to run on a large stack size
    typedef hpx::opencl::server::device_cleanup_action func;
    hpx::async<func>(hpx::find_here(),
                     (uintptr_t)command_queue, (uintptr_t)context).wait();

}

// Initialization function.
// Needed because cl_device_id can not be serialized.
void
device::init(cl_device_id _device_id, bool enable_profiling)
{
    /*
    // TODO remove
    hpx::cout << std::string("Threadsize: ")
              << std::hex << hpx::threads::get_ctx_ptr()->get_stacksize()
              << std::string("\n") << hpx::flush;
    */

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
     *((cl_command_queue_properties *)(supported_queue_properties_data.data()));

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


