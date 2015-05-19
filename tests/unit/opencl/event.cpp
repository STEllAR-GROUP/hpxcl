// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cl_tests.hpp"

#include "../../../opencl/lcos/event.hpp"

void dummy(){}


static void cl_test(hpx::opencl::device cldevice)
{
    typedef hpx::lcos::promise<void> event_type;
    //typedef hpx::opencl::lcos::event<void> event_type;
    typedef typename event_type::wrapped_type event_state_type;



    event_type event;
    hpx::future<void> fut = hpx::async(dummy);//event.get_future();
    hpx::naming::id_type gid = event.get_gid();


    const event_state_type* event2 = static_cast<const event_state_type*>(
                                hpx::lcos::detail::get_shared_state(fut).get());


    hpx::naming::id_type gid2 = event2->get_gid();

    hpx::cout << gid << hpx::endl;
    hpx::cout << gid2 << hpx::endl;
}


