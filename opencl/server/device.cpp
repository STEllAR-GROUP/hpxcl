// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "device.hpp"
#include "../tools.hpp"

#include "buffer.hpp"
#include <CL/cl.h>

//#include <hpx/include/components.hpp>


using hpx::opencl::clx_device_id;
using namespace hpx::opencl::server;

clx_device_id device::test()
{

    return (clx_device_id) NULL;
}


CL_FORBID_EMPTY_CONSTRUCTOR(device);

// Constructor
device::device(clx_device_id _device_id, bool enable_profiling)
{
    this->device_id = (cl_device_id) _device_id;
    
    cl_int err;
    
    // Retrieve platformID
    err = clGetDeviceInfo(this->device_id, CL_DEVICE_PLATFORM,
                          sizeof(platform_id), &platform_id, NULL);
    clEnsure(err, "clGetDeviceInfo()");

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
    clEnsure(err, "clCreateContext()");

    // Create Command Queue
    cl_command_queue_properties command_queue_properties =
                        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    if(enable_profiling)
        command_queue_properties |= CL_QUEUE_PROFILING_ENABLE;
    command_queue = clCreateCommandQueue(context, device_id,
                                         command_queue_properties, &err);
    clEnsure(err, "clCreateCommandQueue()");
}

// Destructor
device::~device()
{
    
    cl_int err;

    // Release command queue
    if(command_queue)
    {
        err = clReleaseCommandQueue(command_queue);
        clEnsure_nothrow(err, "clReleaseCommandQueue()");
        command_queue = NULL; 
    }
    
    // Release context
    if(context)
    {
        err = clReleaseContext(context);
        clEnsure_nothrow(err, "clReleaseContext()");
        context = NULL;
    }

}

hpx::naming::id_type
device::clCreateBuffer(cl_mem_flags flags, size_t size)
{
    hpx::naming::id_type ret = hpx::components::new_<hpx::opencl::server::buffer>
                (hpx::find_here(), (intptr_t) this, flags, size)
                    .get();
    return ret;
}

cl_context
device::getContext()
{
    return context;
}

void CL_CALLBACK
device::error_callback(const char* errinfo, const void* info, size_t info_size,
                                                void* _thisp)
{
    device* thisp = (device*) _thisp;
    hpx::cerr << "device(" << thisp->device_id << "): CONTEXT_ERROR: "
             << errinfo << hpx::endl;
}
