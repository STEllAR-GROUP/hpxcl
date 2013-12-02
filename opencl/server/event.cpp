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

void
event::await() const {
    
    parent_device->wait_for_event(event_id);

}

bool
event::finished() const {

    cl_int err;
    cl_int status;

    // Query the event state
    err = clGetEventInfo(event_id, CL_EVENT_COMMAND_EXECUTION_STATUS,
                           sizeof(cl_int), &status, (size_t *) NULL);
    cl_ensure(err, "clGetEventInfo()");

    // Return true if event is completed
    return status == CL_COMPLETE;

}

boost::shared_ptr<std::vector<char>>
event::get_data()
{

    return parent_device->get_event_data(event_id);

}
