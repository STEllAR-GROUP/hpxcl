// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "cl_tests.hpp"
#include "register_event.hpp"

void cl_test( hpx::opencl::device cldevice, hpx::opencl::device )
{
    typedef hpx::opencl::lcos::event<void> event_type;
    typedef typename event_type::wrapped_type shared_state_type;


    event_type event(cldevice.get_gid());



    auto future = event.get_future();
    auto future_data = hpx::traits::detail::get_shared_state(future);
    auto shared_state = boost::static_pointer_cast<shared_state_type>(future_data);


    auto gid2 = shared_state->get_event_id();
    auto gid = event.get_event_id();
    
    register_event(cldevice, gid);

    HPX_TEST_EQ(gid, gid2);

    future.wait();

    hpx::this_thread::sleep_for(boost::chrono::milliseconds(10));

    hpx::cout << gid << hpx::endl;
    hpx::cout << gid2 << hpx::endl;

}
