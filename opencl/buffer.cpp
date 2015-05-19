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

hpx::future<void>
buffer::enqueue_write_impl( std::size_t offset,
                            std::size_t size,
                            const void* data,
                            std::vector<hpx::naming::id_type> && dependencies)
{
    using hpx::opencl::lcos::event;
    typedef hpx::serialization::serialize_buffer<char> buffer_type;

    // wrap the data in a serialize_buffer object
    buffer_type serializable_data(static_cast<const char*>(data), size,
                                  buffer_type::init_mode::reference);

    // create local event
    event<void> ev(this->get_gid());

    // send command to server class
    typedef hpx::opencl::server::buffer::enqueue_write_action func;
    hpx::apply<func>( this->get_gid(),
                      ev.get_gid(),
                      offset,
                      size,
                      serializable_data,
                      dependencies );
                     

    // return future connected to event
    return ev.get_future();
}
