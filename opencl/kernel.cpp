// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Header File
#include "kernel.hpp"

// Internal Dependencies
#include "server/kernel.hpp"
#include "buffer.hpp"

using hpx::opencl::kernel;

hpx::lcos::future<void>
kernel::set_arg(cl_uint arg_index, const hpx::opencl::buffer &arg) const
{

    HPX_ASSERT(this->get_gid());

    typedef hpx::opencl::server::kernel::set_arg_action func;

    return hpx::async<func>(this->get_gid(), arg_index, arg.get_gid());

}

hpx::future<void>
kernel::enqueue_impl( std::vector<std::size_t> && size_vec,
                      hpx::opencl::util::resolved_events && deps )
{

    // create local event
    using hpx::opencl::lcos::event;
    event<void> ev( device_gid );

    // send command to server class
    typedef hpx::opencl::server::kernel::enqueue_action func;
    hpx::apply<func>( this->get_gid(),
                      ev.get_gid(),
                      size_vec,
                      deps.event_ids );

    // return future connected to event
    return ev.get_future();

}
