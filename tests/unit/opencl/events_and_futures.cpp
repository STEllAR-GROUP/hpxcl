// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cl_tests.hpp"


/*
 * This file tests the event functionality and the conversion event->future and 
 * future->event.
 */

static void cl_test(hpx::opencl::device cldevice)
{
    {
        // Create user event
        hpx::opencl::event user_event = cldevice.create_user_event().get();
    
        // short delay
        sleep(1);
    
        // ensure the event did not trigger yet
        HPX_TEST(user_event.finished().get() == false);
    
        // trigger user event
        user_event.trigger();
    
        // wait for user event to trigger. if test fails, this will deadlock.
        user_event.await();
    
        // ensure the event did now trigger
        HPX_TEST(user_event.finished().get() == true);
    }

    /////////////////////////////////////////////
    // same again, just with a future in between
    /////////////////////////////////////////////

    {
        // Create user event
        hpx::opencl::event user_event = cldevice.create_user_event().get();
    
        // Create a future from the user event
        hpx::lcos::shared_future<void> user_event_future =
                                                        user_event.get_future();

        // Create user event from future
        hpx::opencl::event user_event_future_event =
                    cldevice.create_future_event(user_event_future).get();

        // short delay
        hpx::this_thread::sleep_for(boost::posix_time::milliseconds(100));
    
        // ensure the event did not trigger yet
        HPX_TEST(user_event.finished().get() == false);
        HPX_TEST(user_event_future.is_ready() == false);
        HPX_TEST(user_event_future_event.finished().get() == false);
    
        // trigger user event
        user_event.trigger();
    
        // wait for user event to trigger. if test fails, this will deadlock.
        user_event_future_event.await();
    
        // ensure the event did now trigger
        HPX_TEST(user_event_future_event.finished().get() == true);
        HPX_TEST(user_event_future.is_ready() == true);
        HPX_TEST(user_event.finished().get() == true);
    
    }

}


