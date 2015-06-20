// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_main.hpp>

#include "../../../opencl.hpp"

static void register_event(hpx::opencl::device cldevice,
                           const hpx::naming::id_type & event_id)
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

    parent_device->register_event(event_id, event_cl);
}

#define ATOMIC_PRINT( message )                                                 \
{                                                                               \
    std::stringstream str;                                                      \
    str << "LOG: " << message << std::endl;                                     \
    std::cout << str.str() << std::flush;                                       \
}


int main()
{
        typedef typename hpx::opencl::lcos::event<void>::wrapped_type
            event_type;

        auto devices = 
            hpx::opencl::get_all_devices( CL_DEVICE_TYPE_ALL,
                                          "OpenCL 1.1" ).get();
        HPX_ASSERT(!devices.empty());
        hpx::opencl::device device = devices[0];

        hpx::shared_future<void> fut;
        {
            hpx::opencl::lcos::event<void> ev(device.get_gid());
            register_event(device, ev.get_event_id());
            fut = ev.get_future();
        }

        hpx::this_thread::sleep_for(boost::chrono::milliseconds(10));

        {
            auto shared_state = hpx::lcos::detail::get_shared_state(fut);
            auto ev = boost::static_pointer_cast<event_type>(shared_state);
            ATOMIC_PRINT("ev " << ev->get_event_id());
        }

    
        hpx::this_thread::sleep_for(boost::chrono::milliseconds(10));

        {
            auto shared_state = hpx::lcos::detail::get_shared_state(fut);
            auto ev = boost::static_pointer_cast<event_type>(shared_state);

            ATOMIC_PRINT("trigger_lco_event_manually " << ev->get_event_id());
            hpx::trigger_lco_event(ev->get_event_id());
        }

        ATOMIC_PRINT("before sleep");
        hpx::this_thread::sleep_for(boost::chrono::milliseconds(10));
        ATOMIC_PRINT("after sleep");
        
        {
            auto shared_state = hpx::lcos::detail::get_shared_state(fut);
            auto ev = boost::static_pointer_cast<event_type>(shared_state);
            ATOMIC_PRINT("ev " << ev->get_event_id());
        }

    return 0;
}



