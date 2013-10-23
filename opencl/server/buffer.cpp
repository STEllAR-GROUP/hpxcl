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
    cl_context context = parent_device->get_context();

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

    // Release the device memory
    if(device_mem)
    {
        err = clReleaseMemObject(device_mem);   
        clEnsure(err, "clReleaseMemObject()");
        device_mem = NULL; 
    }
}



// Read Buffer
hpx::opencl::clx_event
buffer::clEnqueueReadBuffer2(size_t offset, size_t size, bool ptr_old,
                            std::vector<clx_event_id> events)
{
    cl_int err;
    cl_event returnEvent;

    // Get the command queue
    cl_command_queue command_queue = parent_device->get_read_command_queue();
    
    // Create the event wait list
    std::vector<cl_event> cl_events_list(events.size());
    cl_event* cl_events_list_ptr = NULL;
    BOOST_FOREACH(clx_event_id & event, events)
    {
        cl_events_list.push_back((cl_event)event);
    }
    if(!cl_events_list.empty())
    {
        cl_events_list_ptr = &cl_events_list[0];
    }
    
    
    // Read the buffer
    char* ptr = new char[size];
    err = clEnqueueReadBuffer(command_queue, device_mem, CL_FALSE, offset,
                              size, (void*)ptr, (cl_uint)events.size(),
                              cl_events_list_ptr, &returnEvent);
    clEnsure(err, "clEnqueueReadBuffer()");
    for(size_t i = 0; i < size; i++)
    {
        std::cout << (int)ptr[i];
    }
    std::cout << std::endl;
    delete[] ptr;

    // Return the clx_event
    return clx_event(parent_device->get_gid(), returnEvent);

}




