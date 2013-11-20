// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "event.hpp"
#include <CL/cl.h>

#include <hpx/include/runtime.hpp>
#include <hpx/util/io_service_pool.hpp>

#include "../tools.hpp"
#include "device.hpp"

using namespace hpx::opencl::server;

CL_FORBID_EMPTY_CONSTRUCTOR(event);

event::event(hpx::naming::id_type device_id, clx_event event_id_)
{
    this->parent_device_id = device_id;
    this->parent_device = hpx::get_ptr
                          <hpx::opencl::server::device>(parent_device_id).get();
    this->event_id = (cl_event) event_id_;
}

event::~event()
{
    // Release ressources in device associated with the event
    parent_device->release_event_resources(event_id);

    // Release the cl_event
    cl_int err;
    err = clReleaseEvent(event_id);
    cl_ensure_nothrow(err, "clReleaseEvent()");
}

cl_event
event::get_cl_event()
{
    return event_id;
}

// Wrapper around the clWaitForEvents.
// This function will be called by an io-threadpool thread, to prevent the
// blocking of an hpx thread (deadlock situation)
static void
hpx_clWaitForEvents(clx_event event_id_,
                    boost::shared_ptr<hpx::lcos::local::promise<cl_int>> p)
{
    // Convert to cl_event
    cl_event event_id = (cl_event)event_id_;
    
    // wait for the given events
    cl_int err = clWaitForEvents(1, &event_id); 

    // return error code via future
    p->set_value(err);
}

void
event::await() const {
    
    // create a promise
    boost::shared_ptr<hpx::lcos::local::promise<int> > p =
        boost::make_shared<hpx::lcos::local::promise<int> >();

    // Get a reference to one of the IO specific HPX io_service objects ...
    hpx::util::io_service_pool* pool =
        hpx::get_runtime().get_thread_pool("io_pool");
    
    // ... and schedule the handler to run the clWaitForEvents on one
    // of its OS-threads.
    pool->get_io_service().post(
    hpx::util::bind(&hpx_clWaitForEvents, (clx_event)event_id, p));

    // wait for it to finish
    cl_int err = p->get_future().get();
    cl_ensure(err, "clWaitForEvents()");

}

boost::shared_ptr<std::vector<char>>
event::get_data()
{

    return parent_device->get_event_data(event_id);

}
