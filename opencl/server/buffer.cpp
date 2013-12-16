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
#include "../buffer.hpp"

using hpx::opencl::server::buffer;
using namespace hpx::opencl::server;

CL_FORBID_EMPTY_CONSTRUCTOR(buffer);


// Constructor
buffer::buffer(hpx::naming::id_type device_id, cl_mem_flags flags, size_t size)
{

    this->parent_device_id = device_id;
    this->parent_device = hpx::get_ptr
                          <hpx::opencl::server::device>(parent_device_id).get();
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
    cl_ensure(err, "clCreateBuffer()");

};

// Constructor
buffer::buffer(hpx::naming::id_type device_id, cl_mem_flags flags, size_t size,
               hpx::util::serialize_buffer<char> data)
{

    BOOST_ASSERT(data.size() == size);

    this->parent_device_id = device_id;
    this->parent_device = hpx::get_ptr
                          <hpx::opencl::server::device>(parent_device_id).get();
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
    cl_ensure(err, "clCreateBuffer()");

};




buffer::~buffer()
{
    // Release the device memory
    if(device_mem)
    {
        parent_device->schedule_cl_mem_deletion(device_mem);
        device_mem = NULL; 
    }
}

// Get Buffer Size
size_t
buffer::size()
{

    size_t size;
    cl_int err;

    // Query size
    err = clGetMemObjectInfo(device_mem, CL_MEM_SIZE, sizeof(size_t), &size,
                                                                          NULL);
    cl_ensure(err, "clGetMemObjectInfo()");

    return size;

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
    cl_ensure(err, "clEnqueueReadBuffer()");

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
    cl_ensure(err, "clEnqueueWriteBuffer()");

    // Register the input data to prevent deallocation
    parent_device->put_event_const_data(returnEvent, data);
    
    // Return the event
    return hpx::opencl::event(
           hpx::components::new_<hpx::opencl::server::event>(
                                hpx::find_here(),
                                parent_device_id,
                                (clx_event) returnEvent
                            ));

}

hpx::opencl::event
buffer::fill(hpx::util::serialize_buffer<char> pattern, size_t offset,
             size_t size, std::vector<hpx::opencl::event> events)
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

    // Fill the buffer
    err = ::clEnqueueFillBuffer(command_queue, device_mem, pattern.data(),
                                pattern.size(), offset, size,
                                (cl_uint)events.size(), cl_events_list_ptr,
                                &returnEvent);

    // Register the input data to prevent deallocation
    parent_device->put_event_const_data(returnEvent, pattern);

    // Return the event
    return hpx::opencl::event(
           hpx::components::new_<hpx::opencl::server::event>(
                                hpx::find_here(),
                                parent_device_id,
                                (clx_event) returnEvent
                            ));

}

hpx::opencl::event
buffer::copy(hpx::naming::id_type src_buffer, std::vector<size_t> dimensions,
             std::vector<hpx::opencl::event> events)
{

    // Parse arguments
    BOOST_ASSERT(dimensions.size() == 3); 
    size_t src_offset = dimensions[0];
    size_t dst_offset = dimensions[1];
    size_t size = dimensions[2];

    // Initialize
    cl_int err;
    cl_event returnEvent;

    // Get buffer locations
    hpx::naming::id_type src_location;
    hpx::naming::id_type dst_location;
    {
        hpx::lcos::future<hpx::naming::id_type>
        src_location_future = get_colocation_id(src_buffer);
        hpx::lcos::future<hpx::naming::id_type>
        dst_location_future = get_colocation_id(get_gid());
        src_location = src_location_future.get();
        dst_location = dst_location_future.get();
    }

    // Decide which way of copying to take
//    if(src_location != dst_location)
    {
        // Brute force copy
        ///////////////////

        // create src buffer client
        hpx::opencl::buffer src(hpx::lcos::make_ready_future(src_buffer));

        // read from src buffer
        hpx::opencl::event read_event = 
                               src.enqueue_read(src_offset, size, events).get();
        
        // transmit the data
        hpx::lcos::future<boost::shared_ptr<std::vector<char>>>
            data_future = read_event.get_data();

        // Get the command queue
        cl_command_queue command_queue = parent_device->get_write_command_queue();

        // Wait for data transmit to finish
        boost::shared_ptr<std::vector<char>> data = data_future.get();

        // write to dst buffer
        err = ::clEnqueueWriteBuffer(command_queue, device_mem, CL_FALSE,
                                     dst_offset, size, &(*data)[0], 0, NULL,
                                     &returnEvent);
        cl_ensure(err, "clEnqueueWriteBuffer()");
        
        // store the data on device as event data to prevent deallocation
        parent_device->put_event_data(returnEvent, data);

    }
        
    
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


