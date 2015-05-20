// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <hpx/config.hpp>
#include <hpx/hpx.hpp>

#include "../../../opencl.hpp"

#include "../../../opencl/lcos/event.hpp"
#include "../../../opencl/server/device.hpp"
#include "../../../opencl/tools.hpp"



template<typename T>
static void register_event(hpx::opencl::device cldevice,
                           const hpx::opencl::lcos::detail::future_base<T> & fut)
{

    boost::shared_ptr<hpx::opencl::server::device>
    parent_device = hpx::get_ptr<hpx::opencl::server::device>
                        (cldevice.get_gid()).get();

    // create a fake event
    cl_int err;
    cl_event event_cl = clCreateUserEvent (
            parent_device->get_context(),
            &err);
    cl_ensure(err, "clEnqueueWriteBuffer()");
    err = clSetUserEventStatus(event_cl, CL_COMPLETE);
    cl_ensure(err, "clSetUserEventStatus()");

    parent_device->register_event(fut.get_event_id(), event_cl);
}

