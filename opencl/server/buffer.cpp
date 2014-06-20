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
#include "../device.hpp"

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
        cl_events_list_ptr = cl_events_list.data();
    }

    // Create the buffer
    boost::shared_ptr<std::vector<char>> buffer = 
                                    boost::make_shared<std::vector<char>>(size);

    // Read the buffer
    err = ::clEnqueueReadBuffer(command_queue, device_mem, CL_FALSE, offset,
                              size, (void*)(buffer->data()), (cl_uint)events.size(),
                              cl_events_list_ptr, &returnEvent);
    cl_ensure(err, "clEnqueueReadBuffer()");

    // Send buffer to device class
    parent_device->put_event_data(returnEvent, buffer);
    
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
        cl_events_list_ptr = cl_events_list.data();
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

#ifdef CL_VERSION_1_2
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
        cl_events_list_ptr = cl_events_list.data();
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
#endif

// Bruteforce copy, needed for copy between different machines
cl_event
buffer::copy_bruteforce(hpx::naming::id_type & src_buffer,
                        const size_t & src_offset,
                        const size_t & dst_offset,
                        const size_t & size,
                        std::vector<hpx::opencl::event> & events)
{
        cl_int err;
        cl_event returnEvent;

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
                                     dst_offset, size, data->data(), 0, NULL,
                                     &returnEvent);
        cl_ensure(err, "clEnqueueWriteBuffer()");
        
        // store the data on device as event data to prevent deallocation
        parent_device->put_event_data(returnEvent, data);

        // return the event
        return returnEvent;
}

// Local copy, same process but different context
cl_event
buffer::copy_local(boost::shared_ptr<hpx::opencl::server::buffer> src,
                   const size_t & src_offset,
                   const size_t & dst_offset,
                   const size_t & size,
                   std::vector<hpx::opencl::event> & events)
{
        cl_int err;
        cl_event returnEvent;

        // Get the cl_event dependency list
        std::vector<cl_event> cl_events_list = hpx::opencl::event::
                                                        get_cl_events(events);
        cl_event* cl_events_list_ptr = NULL;
        if(!cl_events_list.empty())
        {
            cl_events_list_ptr = cl_events_list.data();
        }

        // create a copy buffer
        boost::shared_ptr<std::vector<char>> copy_buffer = 
                                    boost::make_shared<std::vector<char>>(size);
        
        // get the read command queue
        cl_command_queue src_command_queue =
                                   src->parent_device->get_read_command_queue();

        // Read into buffer
        cl_event read_event_;
        err = ::clEnqueueReadBuffer(src_command_queue, src->device_mem,
                                    CL_FALSE, src_offset, size,
                                    (void*)(copy_buffer->data()),
                                    (cl_uint)events.size(),
                                    cl_events_list_ptr, &read_event_);
        cl_ensure(err, "clEnqueueReadBuffer()");

        // Create hpx::opencl::event from cl_event
        hpx::opencl::event read_event(
               hpx::components::new_<hpx::opencl::server::event>(
                                    hpx::find_here(),
                                    src->parent_device_id,
                                    (clx_event) read_event_
                                ));

        // Create future from event
        hpx::lcos::future<void> read_future = read_event.get_future();

        // Create device client of dst
        hpx::opencl::device dst_device(
                                hpx::lcos::make_ready_future(parent_device_id));

        // Create new event on dst device
        hpx::opencl::event write_start_event_client = 
                   dst_device.create_future_event(std::move(read_future)).get();

        // Convert dst event to cl_event
        cl_event write_start_event =
                     hpx::opencl::event::get_cl_event(write_start_event_client);

        // get write command queue
        cl_command_queue dst_command_queue = 
                                       parent_device->get_write_command_queue();

        // Write to device
        err = ::clEnqueueWriteBuffer(dst_command_queue, device_mem, CL_FALSE,
                                     dst_offset, size, copy_buffer->data(), 
                                     1, &write_start_event,
                                     &returnEvent);
        cl_ensure(err, "clEnqueueWriteBuffer()");
        
        // Send buffer to device class to prevent deallocation
        parent_device->put_event_data(returnEvent, copy_buffer);
 
        // return the event
        return returnEvent;
}

// Direct copy, buffers are on the same context
cl_event
buffer::copy_direct(boost::shared_ptr<hpx::opencl::server::buffer> src,
                    const size_t & src_offset,
                    const size_t & dst_offset,
                    const size_t & size,
                    std::vector<hpx::opencl::event> & events)
{
        cl_int err;
        cl_event returnEvent;

        // Get the cl_event dependency list
        std::vector<cl_event> cl_events_list = hpx::opencl::event::
                                                        get_cl_events(events);
        cl_event* cl_events_list_ptr = NULL;
        if(!cl_events_list.empty())
        {
            cl_events_list_ptr = cl_events_list.data();
        }

        // get command queue
        cl_command_queue command_queue = 
                                       parent_device->get_write_command_queue();

        // Perform direct copy
        err = ::clEnqueueCopyBuffer(command_queue, src->device_mem, device_mem,
                                    src_offset, dst_offset, size, 
                                    (cl_uint)events.size(),
                                    cl_events_list_ptr, &returnEvent);
        cl_ensure(err, "clEnqueueCopyBuffer()");
       
        // return the evetn
        return returnEvent;
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
    if(src_location != dst_location)
    {
        // Data is on different machines/processes, brute force copy.
        returnEvent = copy_bruteforce(src_buffer, src_offset, dst_offset,
                                      size, events);
    }
    else
    {
        // Data is on the same machine and process.
        boost::shared_ptr<hpx::opencl::server::buffer> src = 
                    hpx::get_ptr<hpx::opencl::server::buffer>(src_buffer).get();
        cl_context src_context = src->parent_device->get_context();
        cl_context dst_context = this->parent_device->get_context();

        if(src_context != dst_context)
        {
            // Data is on the same process, but on different contexts
            returnEvent = copy_local(src, src_offset, dst_offset, size, events);
        }
        else
        {
            // Data is on the same context, perform direct copy
            returnEvent = copy_direct(src, src_offset, dst_offset, size, events);
        }

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


