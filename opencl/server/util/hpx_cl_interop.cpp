// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "hpx_cl_interop.hpp"

#include <hpx/runtime.hpp>

#include <atomic>

// This is the number of an OpenCL thread.
// It gets increased with every OpenCL call, to prevent name collisions
static std::atomic<std::size_t> opencl_thread_num(0);

// This function triggers an hpx::lcos::local::event from an external thread
void
hpx::opencl::server::util
::set_promise_from_external( hpx::runtime * rt,
                             hpx::lcos::local::promise<cl_int> * promise,
                             cl_int value )
{

    // If we are on an hpx thread we don't need any special treatment
    if(rt->get_thread_name() != "<unknown>")
    {
        promise->set_value(value);
        return;
    }

    // if we're on an OS thread, register it temporarily.
    // add the thread id to its name, as there could potentially
    // be multiple OpenCL threads in this function at the same time
    rt->register_thread("opencl",
                        opencl_thread_num.fetch_add(1,
                                                    std::memory_order_relaxed),
                        false);
    //BOOST_ASSERT(succeeded);

    // trigger the event lock
    promise->set_value(value);

    // unregister the thread from hpx as we don't have any control over it
    // any more. ever. (probably)
    // /* this line is currently commented out.
    //  * is unregistering necessary?
    //  * there should be huge speed improvements if we don't unregister.
    //  * although it would be a potential memory leak.
    //  * but it would be kind of a memory leak as well, if we register
    //  * every single callback-call on a different thread name ...
    //  */
    //rt->unregister_thread();

}

// This function triggers an hpx::lcos::local::event from an external thread
void
hpx::opencl::server::util
::set_promise_from_external( hpx::runtime * rt,
                             hpx::lcos::local::promise<void> * promise )
{

    // If we are on an hpx thread we don't need any special treatment
    if(rt->get_thread_name() != "<unknown>")
    {
        promise->set_value();
        return;
    }

    // if we're on an OS thread, register it temporarily.
    // add the thread id to its name, as there could potentially
    // be multiple OpenCL threads in this function at the same time
    rt->register_thread("opencl",
                        opencl_thread_num.fetch_add(1,
                                                    std::memory_order_relaxed),
                        false);
    //BOOST_ASSERT(succeeded);

    // trigger the event lock
    promise->set_value();

    // unregister the thread from hpx as we don't have any control over it
    // any more. ever. (probably)
    // /* this line is currently commented out.
    //  * is unregistering necessary?
    //  * there should be huge speed improvements if we don't unregister.
    //  * although it would be a potential memory leak.
    //  * but it would be kind of a memory leak as well, if we register
    //  * every single callback-call on a different thread name ...
    //  */
    //rt->unregister_thread();

}
