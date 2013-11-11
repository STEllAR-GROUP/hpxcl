// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>
#include <hpx/runtime/get_ptr.hpp>

#include <CL/cl.h>

#include "buffer.hpp"
#include "../tools.hpp"
#include "device.hpp"
#include "../event.hpp"

using hpx::opencl::server::buffer;
using namespace hpx::opencl::server;

CL_FORBID_EMPTY_CONSTRUCTOR(buffer);


// Constructor
buffer::buffer(hpx::naming::id_type device_id, cl_mem_flags flags, size_t size_)
{

    this->parent_device_id = device_id;
    this->parent_device = hpx::get_ptr
                          <hpx::opencl::server::device>(parent_device_id).get();
    this->size = size_;
    this->device_mem = NULL;

    // Retrieve the context from parent class
    cl_context context = parent_device->get_context();

    // The opencl error variable
    cl_int err;

    // Modify the cl_mem_flags
    cl_mem_flags modified_flags = flags &! (CL_MEM_USE_HOST_PTR
                                            | CL_MEM_ALLOC_HOST_PTR
                                            | CL_MEM_COPY_HOST_PTR);
    
    // Create the Context
    device_mem = clCreateBuffer(context, modified_flags, size, NULL, &err);
    clEnsure(err, "clCreateBuffer()");

};

// Constructor
buffer::buffer(hpx::naming::id_type device_id, cl_mem_flags flags, size_t size_,
               hpx::util::serialize_buffer<char> data)
{

    BOOST_ASSERT(data.size() == size_);

    this->parent_device_id = device_id;
    this->parent_device = hpx::get_ptr
                          <hpx::opencl::server::device>(parent_device_id).get();
    this->size = size_;
    this->device_mem = NULL;

    // Retrieve the context from parent class
    cl_context context = parent_device->get_context();

    // The opencl error variable
    cl_int err;

    // Modify the cl_mem_flags
    cl_mem_flags modified_flags = flags &! (CL_MEM_USE_HOST_PTR
                                            | CL_MEM_ALLOC_HOST_PTR);
    modified_flags = modified_flags | CL_MEM_COPY_HOST_PTR; 

    // Create the Context
    device_mem = clCreateBuffer(context, modified_flags, size,
                                          const_cast<char*>(data.data()), &err);
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
hpx::opencl::event
buffer::read(size_t offset, size_t size,
                            std::vector<hpx::opencl::event> events)
{
    cl_int err;
    cl_event returnEvent;

    // Get the command queue
    cl_command_queue command_queue = parent_device->get_read_command_queue();
    
    
    // Get the cl_event dependency list
    std::vector<cl_event> cl_events_list = hpx::opencl::event::
                                                    get_cl_events(events);
    cl_event* cl_events_list_ptr = NULL;
    if(!cl_events_list.empty())
    {
        cl_events_list_ptr = &cl_events_list[0];
    }

    // Create the buffer
    std::vector<char> *buffer = new std::vector<char>(size);
    boost::shared_ptr<std::vector<char>> buffer_ptr(buffer);

    // Read the buffer
    err = ::clEnqueueReadBuffer(command_queue, device_mem, CL_FALSE, offset,
                              size, (void*)&(*buffer)[0], (cl_uint)events.size(),
                              cl_events_list_ptr, &returnEvent);
    clEnsure(err, "clEnqueueReadBuffer()");

    // Send buffer to device class
    parent_device->put_event_data(returnEvent, buffer_ptr);
    
    // Return the event
    return hpx::opencl::event(
           hpx::components::new_<hpx::opencl::server::event>(
                                hpx::find_here(),
                                parent_device_id,
                                (clx_event) returnEvent
                            ));

}

hpx::opencl::event
buffer::write(size_t offset, hpx::util::serialize_buffer<char> data,
                             std::vector<hpx::opencl::event> events)
{
    
    cl_int err;
    cl_event returnEvent;

    // Get the command queue
    cl_command_queue command_queue = parent_device->get_write_command_queue();
    
    
    // Get the cl_event dependency list
    std::vector<cl_event> cl_events_list = hpx::opencl::event::
                                                    get_cl_events(events);
    cl_event* cl_events_list_ptr = NULL;
    if(!cl_events_list.empty())
    {
        cl_events_list_ptr = &cl_events_list[0];
    }

    // Write to the buffer
    err = ::clEnqueueWriteBuffer(command_queue, device_mem, CL_FALSE, offset,
                                 data.size(), data.data(), (cl_uint)events.size(),
                                 cl_events_list_ptr, &returnEvent);
    clEnsure(err, "clEnqueueWriteBuffer()");
    
    // Return the event
    return hpx::opencl::event(
           hpx::components::new_<hpx::opencl::server::event>(
                                hpx::find_here(),
                                parent_device_id,
                                (clx_event) returnEvent
                            ));

}

cl_mem
buffer::get_cl_mem()
{

    return device_mem;

}


