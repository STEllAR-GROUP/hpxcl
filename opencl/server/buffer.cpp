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

using hpx::opencl::server::buffer;
using hpx::lcos::future;
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
hpx::opencl::event
buffer::clEnqueueReadBuffer(size_t offset, size_t size,
                            std::vector<hpx::opencl::event> events)
{
    cl_int err;
    cl_event returnEvent;

    // Get the command queue
    cl_command_queue command_queue = parent_device->get_read_command_queue();
   /* 
    // Fetch opencl event component pointers
    std::vector<future<boost::shared_ptr<hpx::opencl::server::event>>>
            event_server_futures(events.size());
    BOOST_FOREACH(hpx::opencl::event & event, events)
    {
        event_server_futures.push_back(
            hpx::get_ptr<hpx::opencl::server::event>(event));
    }
    // Wait for event component fetching to finish
    std::vector<boost::shared_ptr<hpx::opencl::server::event>>
            event_servers(events.size());
    BOOST_FOREACH(future<boost::shared_ptr<hpx::opencl::server::event>>
                        & event_server_future, event_server_futures)
    {
        event_servers.push_back(event_server_future.get());
    }

    // Fetch the cl_event pointers from event servers and create eventlist
    std::vector<cl_event> cl_events_list(events.size());
    BOOST_FOREACH(boost::shared_ptr<hpx::opencl::server::event> & event_server,
                                                                event_servers)
    {
        cl_events_list->push_back(event.get_cl_event());
    }
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
    for(size_t i = 0; i < size; i++)
    {
        std::cout << (int)(*buffer)[i];
    }
    std::cout << std::endl;

    // Add buffer to read map
    // lock
    read_map.insert(
        std::pair<clx_event_id, boost::shared_ptr<std::vector<char>>>
            ((clx_event_id)returnEvent, buffer_ptr));
    // unlock
    
    // Return the clx_event
    return clx_event(parent_device->get_gid(), returnEvent);
*/
   
    return hpx::opencl::event();
}




