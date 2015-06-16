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
    cl_int err;

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    cl_mem device_mem = reinterpret_cast<cl_mem>(device_mem_ptr);

    // Release the device memory
    if(device_mem)
    {
        err = clReleaseMemObject(device_mem);
        cl_ensure_nothrow(err, "clReleaseMemObject()");
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

    this->parent_device_id = std::move(device_id);
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
    cl_ensure(err, "clEnqueueReadBuffer()");

    // register the data to prevent deallocation
    parent_device->put_event_data(return_event, data);

    // register the cl_event to the client event
    parent_device->register_event(event_gid, return_event);

    // arm the future. ! this blocks.
    parent_device->activate_deferred_event_with_data(event_gid);

}


void
buffer::send_bruteforce( hpx::naming::id_type && dst,
                         hpx::naming::id_type && src_event_gid,
                         hpx::naming::id_type && dst_event_gid,
                         std::size_t src_offset,
                         std::size_t dst_offset,
                         std::size_t size,
                         std::vector<hpx::naming::id_type> && src_dependencies,
                         std::vector<hpx::naming::id_type> && dst_dependencies)
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 
    
    ////////////////////////////////////////////////////////////////////////////
    // Read
    //

    typedef hpx::serialization::serialize_buffer<char> buffer_type;

    cl_int err;
    cl_event src_event;

    // retrieve the dependency cl_events
    util::event_dependencies events( src_dependencies, parent_device.get() );

    // retrieve the command queue
    cl_command_queue command_queue = parent_device->get_read_command_queue();

    // create new target buffer
    buffer_type data( new char[size], size, buffer_type::init_mode::take );

    // run the OpenCL-call
    err = clEnqueueReadBuffer( command_queue, device_mem, CL_FALSE, src_offset,
                                data.size(), data.data(),
                                static_cast<cl_uint>(events.size()),
                                events.get_cl_events(), &src_event );
    cl_ensure(err, "clEnqueueReadBuffer()");

    // register the cl_event to the client event
    parent_device->register_event(src_event_gid, src_event);

    // wait for clEnqueueReadBuffer to finish
    parent_device->wait_for_cl_event(src_event);

    ////////////////////////////////////////////////////////////////////////////
    // Write
    //
    typedef hpx::opencl::server::buffer::enqueue_write_action<char> func;
    hpx::apply<func>( std::move(dst),
                      std::move(dst_event_gid),
                      dst_offset,
                      data,
                      std::move(dst_dependencies) );
}

void
buffer::send_direct( hpx::naming::id_type && dst,
                     boost::shared_ptr<hpx::opencl::server::buffer> && dst_buffer,
                     hpx::naming::id_type && src_event_gid,
                     hpx::naming::id_type && dst_event_gid,
                     std::size_t src_offset,
                     std::size_t dst_offset,
                     std::size_t size,
                     std::vector<hpx::naming::id_type> && src_dependencies,
                     std::vector<hpx::naming::id_type> && dst_dependencies)
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 
    
    cl_int err;
    cl_event return_event;

    // gather all dependencies from both devices
    std::vector<cl_event> events;
    events.reserve(src_dependencies.size() + dst_dependencies.size());
    for(const auto& id : src_dependencies){
        events.push_back( parent_device->retrieve_event(id) );
    }
    for(const auto& id : dst_dependencies){
        events.push_back( dst_buffer->parent_device->retrieve_event(id) );
    }

    // Create a pointer that is either a pointer to the data or NULL
    cl_event* events_ptr = NULL;
    if(!events.empty()){
        events_ptr = events.data();
    }

    // retrieve the command queue
    cl_command_queue command_queue = parent_device->get_write_command_queue();

    // run the OpenCL-call
    err = clEnqueueCopyBuffer( command_queue, device_mem, dst_buffer->device_mem,
                               src_offset, dst_offset, size,
                               static_cast<cl_uint>(events.size()),
                               events_ptr, &return_event );
    cl_ensure(err, "clEnqueueCopyBuffer()");

    // retain event to enable double-registration
    err = clRetainEvent( return_event );
    cl_ensure(err, "clRetainEvent()");

    // register the cl_event to both client events
    this->parent_device->register_event(src_event_gid, return_event);
    dst_buffer->parent_device->register_event(dst_event_gid, return_event);

}

void
buffer::enqueue_send( hpx::naming::id_type && dst,
                      hpx::naming::id_type && src_event,
                      hpx::naming::id_type && dst_event,
                      std::size_t src_offset,
                      std::size_t dst_offset,
                      std::size_t size,
                      std::vector<hpx::naming::id_type> && dependencies,
                      std::vector<hpx::naming::gid_type> && dependency_devices )
{

    HPX_ASSERT(dependencies.size() == dependency_devices.size());

    // query the location of the destination
    auto dst_location_future = hpx::get_colocation_id(dst);

    // split between src_dependencies and dst_dependencies
    std::vector<hpx::naming::id_type> src_dependencies;
    std::vector<hpx::naming::id_type> dst_dependencies;
    hpx::naming::gid_type src_device = parent_device_id.get_gid();
    std::vector<hpx::naming::id_type>::iterator it = dependencies.begin();
    for(const auto& device : dependency_devices){
        if(device == src_device){
            std::move(it, it+1, std::back_inserter(src_dependencies));
        } else {
            std::move(it, it+1, std::back_inserter(dst_dependencies));
        }
        it++;
    }

    // get the location of the destination
    hpx::naming::id_type dst_location = dst_location_future.get();
    hpx::naming::id_type src_location = hpx::find_here();

    // choose which function to run
    // optimization for context internal copies
    if(dst_location == src_location){
        auto dst_buffer = hpx::get_ptr<hpx::opencl::server::buffer>(dst).get();

        cl_context src_context = this->parent_device->get_context();
        cl_context dst_context = dst_buffer->parent_device->get_context();

        if(src_context == dst_context){
            send_direct( std::move(dst),
                         std::move(dst_buffer),
                         std::move(src_event),
                         std::move(dst_event),
                         src_offset,
                         dst_offset,
                         size,
                         std::move(src_dependencies),
                         std::move(dst_dependencies) );
            return;
        }
    }

    // Always works: the bruteforce method
    send_bruteforce( std::move(dst),
                     std::move(src_event),
                     std::move(dst_event),
                     src_offset,
                     dst_offset,
                     size,
                     std::move(src_dependencies),
                     std::move(dst_dependencies) );

}


