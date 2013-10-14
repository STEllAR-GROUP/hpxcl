// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "memory.hpp"
#include "../tools.hpp"

#include <CL/cl.h>


using hpx::opencl::clx_device_id;
using namespace hpx::opencl::server;

CL_FORBID_EMPTY_CONSTRUCTOR(memory);

// Constructor
memory::memory(device* parent_device, size_t size)
{
    this->size = size;
    this->parent_device = parent_device;
    
    // Initialize host memory, will be pinned and used for memory mapping
    host_mem = std::vector<char>(size);

    // Don't initialize device memory. Will get initialized by derived classes.
    device_mem = NULL;
}

// Destructor
memory::~memory()
{

    cl_int err;

    // Release device memory
    if(device_mem)
    {
        err = clReleaseMemObject(device_mem);
        clEnsure_nothrow(err, "clReleaseMemObject()");
        device_mem = NULL;
    }
}

