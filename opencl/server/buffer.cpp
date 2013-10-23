// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>

#include <CL/cl.h>

#include "buffer.hpp"
#include "../tools.hpp"
#include "device.hpp"

using hpx::opencl::server::buffer;
using namespace hpx::opencl::server;


CL_FORBID_EMPTY_CONSTRUCTOR(buffer);


// Constructor
buffer::buffer(intptr_t _parent_device, cl_mem_flags flags, size_t _size,
               char* init_data)
{

    this->parent_device = (device*) _parent_device;
    this->size = _size;
    this->device_mem = NULL;

    // Retrieve the context from parent class
    cl_context context = parent_device->getContext();

    // The opencl error variable
    cl_int err;

    // Modify the cl_mem_flags
    cl_mem_flags modified_flags = flags &! (CL_MEM_USE_HOST_PTR
                                            || CL_MEM_ALLOC_HOST_PTR);

    // Create the Context
    device_mem = clCreateBuffer(context, modified_flags, size, init_data, &err);
    clEnsure(err, "clCreateBuffer()");


};


buffer::~buffer()
{
    cl_int err;
    if(device_mem)
    {
        err = clReleaseMemObject(device_mem);   
        clEnsure(err, "clReleaseMemObject()");
        device_mem = NULL; 
    }
}








