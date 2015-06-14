// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Header File
#include "buffer.hpp"

// Internal Dependencies
#include "server/buffer.hpp"

#include "lcos/event.hpp"

using hpx::opencl::buffer;


hpx::future<std::size_t>
buffer::size() const
{
    
    HPX_ASSERT(this->get_gid());
    typedef hpx::opencl::server::buffer::size_action func;

    return hpx::async<func>(this->get_gid());

}

hpx::future<hpx::serialization::serialize_buffer<char> >
buffer::enqueue_read_alloc_impl(
    std::size_t offset,
    std::size_t size,
    hpx::opencl::util::resolved_events && dependencies )
{
    using hpx::opencl::lcos::event;
    typedef hpx::serialization::serialize_buffer<char> buffer_type;

    HPX_ASSERT(dependencies.are_from_device(device_gid));

    // create local event
    event<buffer_type> ev( device_gid );
   
    // send command to server class
    typedef hpx::opencl::server::buffer::enqueue_read_action func;
    hpx::apply<func>( this->get_gid(),
                      ev.get_gid(),
                      offset,
                      size,
                      dependencies.event_ids ); 

    // return future connected to event
    return ev.get_future();
}
