// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The Header of this class
#include "buffer.hpp"

// HPXCL tools
#include "../tools.hpp"

// other hpxcl dependencies
#include "device.hpp"
#include "util/event_dependencies.hpp"

// HPX dependencies
#include <hpx/include/thread_executors.hpp>


using namespace hpx::opencl::server;


// Constructor
buffer::buffer()
{}

// External destructor.
// This is needed because OpenCL calls only run properly on large stack size.
static void buffer_cleanup(uintptr_t device_mem_ptr)
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    cl_mem device_mem = reinterpret_cast<cl_mem>(device_mem_ptr);

    // Release the device memory
    if(device_mem)
    {
         clReleaseMemObject(device_mem);
    }
}

// Destructor
buffer::~buffer()
{

    hpx::threads::executors::default_executor exec(
                                          hpx::threads::thread_priority_normal,
                                          hpx::threads::thread_stacksize_large);

    // run dectructor in a thread, as we need it to run on a large stack size
    hpx::async( exec, &buffer_cleanup, reinterpret_cast<uintptr_t>(device_mem))
                                                                        .wait();


}


void
buffer::init( hpx::naming::id_type device_id, cl_mem_flags flags,
                                              std::size_t size)
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

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

}

// Get Buffer Size
std::size_t
buffer::size()
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    std::size_t size;
    cl_int err;

    // Query size
    err = clGetMemObjectInfo(device_mem, CL_MEM_SIZE, sizeof(std::size_t), &size,
                                                                          NULL);
    cl_ensure(err, "clGetMemObjectInfo()");

    return size;

}

void
buffer::enqueue_write( hpx::naming::id_type && event_gid,
                       std::size_t offset,
                       std::size_t size,
                       hpx::serialization::serialize_buffer<char> data,
                       std::vector<hpx::naming::id_type> && dependencies ){
    
    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    cl_int err;
    cl_event return_event;

    // retrieve the dependency cl_events
    util::event_dependencies events( dependencies, parent_device.get() );

    // retrieve the command queue
    cl_command_queue command_queue = parent_device->get_write_command_queue();

    // run the OpenCL-call
    err = clEnqueueWriteBuffer( command_queue, device_mem, CL_FALSE, offset,
                                data.size(), data.data(),
                                static_cast<cl_uint>(events.size()),
                                events.get_cl_events(), &return_event );
    cl_ensure(err, "clEnqueueWriteBuffer()");

    // register the data to prevent deallocation
    parent_device->put_event_data(return_event, data);

    // register the cl_event to the client event
    parent_device->register_event(event_gid, return_event);

}

void
buffer::enqueue_read( hpx::naming::id_type && event_gid,
                      std::size_t offset,
                      std::size_t size,
                      std::vector<hpx::naming::id_type> && dependencies ){

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 
    
    typedef hpx::serialization::serialize_buffer<char> buffer_type;

    cl_int err;
    cl_event return_event;

    // retrieve the dependency cl_events
    util::event_dependencies events( dependencies, parent_device.get() );

    // retrieve the command queue
    cl_command_queue command_queue = parent_device->get_read_command_queue();

    // create new target buffer
    buffer_type data( new char[size], size, buffer_type::init_mode::take );

    // run the OpenCL-call
    err = clEnqueueReadBuffer( command_queue, device_mem, CL_FALSE, offset,
                                data.size(), data.data(),
                                static_cast<cl_uint>(events.size()),
                                events.get_cl_events(), &return_event );
    cl_ensure(err, "clEnqueueWriteBuffer()");

    // register the data to prevent deallocation
    parent_device->put_event_data(return_event, data);

    // register the cl_event to the client event
    parent_device->register_event(event_gid, return_event);

    // arm the future
    parent_device->activate_deferred_event_with_data(event_gid);

}


