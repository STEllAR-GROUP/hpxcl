// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "device.hpp"
#include "../tools.hpp"

#include "buffer.hpp"
#include <CL/cl.h>

#include <boost/foreach.hpp>

//#include <hpx/include/components.hpp>


using hpx::opencl::clx_device_id;
using namespace hpx::opencl::server;

CL_FORBID_EMPTY_CONSTRUCTOR(device);

// Constructor
device::device(clx_device_id _device_id, bool enable_profiling)
{
    this->device_id = (cl_device_id) _device_id;
    
    cl_int err;
    
    // Retrieve platformID
    err = clGetDeviceInfo(this->device_id, CL_DEVICE_PLATFORM,
                          sizeof(platform_id), &platform_id, NULL);
    cl_ensure(err, "clGetDeviceInfo()");

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
    cl_ensure(err, "clCreateContext()");

    // Create Command Queue
    cl_command_queue_properties command_queue_properties =
                        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    if(enable_profiling)
        command_queue_properties |= CL_QUEUE_PROFILING_ENABLE;
    command_queue = clCreateCommandQueue(context, device_id,
                                         command_queue_properties, &err);
    cl_ensure(err, "clCreateCommandQueue()");
}

// Destructor
device::~device()
{
    cl_int err;

    // cleanup user events and pending cl_mem deletions
    cleanup_user_events();

    // Release command queue
    if(command_queue)
    {
        err = clFinish(command_queue);
        cl_ensure_nothrow(err, "clFinish()");
        err = clReleaseCommandQueue(command_queue);
        cl_ensure_nothrow(err, "clReleaseCommandQueue()");
        command_queue = NULL; 
    }
    
    // Release context
    if(context)
    {
        err = clReleaseContext(context);
        cl_ensure_nothrow(err, "clReleaseContext()");
        context = NULL;
    }
    
}


cl_context
device::get_context()
{
    return context;
}

cl_device_id
device::get_device_id()
{
    return device_id;
}

cl_command_queue
device::get_read_command_queue()
{
    return command_queue;
}

cl_command_queue
device::get_write_command_queue()
{
    return command_queue;
}

cl_command_queue
device::get_work_command_queue()
{
    return command_queue;
}

void
device::put_event_data(cl_event ev, boost::shared_ptr<std::vector<char>> mem)
{
    
    // Insert buffer to buffer map
    boost::lock_guard<hpx::lcos::local::mutex> lock(event_data_mutex);
    event_data.insert(
            std::pair<cl_event, boost::shared_ptr<std::vector<char>>>
                        (ev, mem));
    
}

void
device::release_event_resources(cl_event event_id)
{
   
    // Wait for events to end
    clWaitForEvents(1, &event_id);
   
    // Delete all associated read buffers
    boost::lock_guard<hpx::lcos::local::mutex> lock(event_data_mutex);
    event_data.erase(event_id);
    
}

boost::shared_ptr<std::vector<char>>
device::get_event_data(cl_event event)
{

    // wait for event to finish
    clWaitForEvents(1, &event);

    // synchronization
    boost::lock_guard<hpx::lcos::local::mutex> lock(event_data_mutex);

    // retrieve the data
    std::map<cl_event, boost::shared_ptr<std::vector<char>>>::iterator
    it = event_data.find(event);

    // Check for object exists. Should exist in a bug-free program.
    BOOST_ASSERT (it != event_data.end());

    // Return the data pointer
    return it->second;

}

hpx::opencl::event
device::create_user_event()
{

    BOOST_ASSERT(this->get_gid());

    cl_int err;
    cl_event event;
   
    // lock user_events_mutex, so no cl_mem can be deleted from now on
    boost::lock_guard<hpx::lcos::local::mutex> lock(user_events_mutex);

    // create the user event
    event = clCreateUserEvent(context, &err);
    cl_ensure(err, "clCreateUserEvent()");

    // initialize cl_event component client
    hpx::opencl::event event_client(
                       hpx::components::new_<hpx::opencl::server::event>(
                                            hpx::find_here(),
                                            get_gid(),
                                            (clx_event) event
                                        ));

    // add event to list of user events
    user_events.insert(
            std::pair<cl_event, hpx::opencl::event>(event, event_client)
                            );

    // Return the event
    return event_client;

}


void
device::trigger_user_event(hpx::opencl::event event)
{

    // lock user_events_mutex
    boost::lock_guard<hpx::lcos::local::mutex> lock(user_events_mutex);
   
    trigger_user_event_nolock(event);

}

void
device::trigger_user_event_nolock(hpx::opencl::event event)
{

    // this function assumes that user_events_mutex is already locked
    
    // convert event to cl_event
    cl_event event_ = hpx::opencl::event::get_cl_event(event);

    // check if event is a user event on this device
    if(user_events.erase(event_) < 1)
        return;

    // trigger event
    cl_int err;
    err = clSetUserEventStatus(event_, CL_COMPLETE);
    cl_ensure(err, "clSetUserEventStatus()");

    // try to delete cl_mems.
    // call nolock version, as user_events_mutex is already locked.
    try_delete_cl_mem_nolock();    

}

void
device::schedule_cl_mem_deletion(cl_mem mem)
{
    
    // add mem to list of pending deletions
    pending_cl_mem_deletions_mutex.lock();
    pending_cl_mem_deletions.push(mem);
    pending_cl_mem_deletions_mutex.unlock();

    // try to delete it
    try_delete_cl_mem();

}

void
device::try_delete_cl_mem()
{
    
    // lock user_events
    boost::lock_guard<hpx::lcos::local::mutex> lock(user_events_mutex);

    // call nolock function, as we just locked user_events_mutex.
    try_delete_cl_mem_nolock();

}

void
device::try_delete_cl_mem_nolock()
{

    // this function assumes that user_events_mutex is already locked externally.

    // if still user events pending, don't delete
    if(!user_events.empty())
        return;

    // else: delete. keep user_events_mutex locked, so no new events can be
    // generated while deletion.

    // lock pending_cl_mem_deletions
    boost::lock_guard<hpx::lcos::local::mutex> lock(pending_cl_mem_deletions_mutex);

    // delete cl_mems
    while(!pending_cl_mem_deletions.empty())
    {
        ::clReleaseMemObject(pending_cl_mem_deletions.front());
        pending_cl_mem_deletions.pop();
    }

}

void
device::cleanup_user_events()
{

    // trigger all user generated events
    boost::lock_guard<hpx::lcos::local::mutex> lock(user_events_mutex);
    std::vector<hpx::opencl::event> leftover_user_events;
    leftover_user_events.reserve(user_events.size());
    typedef std::map<cl_event, hpx::opencl::event> user_events_map_type;
    BOOST_FOREACH(user_events_map_type::value_type &user_event,
                  user_events)
    {
        leftover_user_events.push_back(user_event.second);
    }
    BOOST_FOREACH(hpx::opencl::event & leftover_user_event, leftover_user_events)
    {
        trigger_user_event_nolock(leftover_user_event);
    }

    // Release all leftover cl_mems
    try_delete_cl_mem_nolock();
    boost::lock_guard<hpx::lcos::local::mutex> lock2(pending_cl_mem_deletions_mutex);
    if(pending_cl_mem_deletions.size() > 0)
        hpx::cerr << "Unable to release all cl_mem objects!" << hpx::endl;

}

void CL_CALLBACK
device::error_callback(const char* errinfo, const void* info, size_t info_size,
                                                void* _thisp)
{
    device* thisp = (device*) _thisp;
    hpx::cerr << "device(" << thisp->device_id << "): CONTEXT_ERROR: "
             << errinfo << hpx::endl;
}



