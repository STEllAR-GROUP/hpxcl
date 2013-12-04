// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include "server/buffer.hpp"

#include "buffer.hpp"

using hpx::opencl::buffer;


hpx::lcos::future<size_t>
buffer::size() const
{
    
    BOOST_ASSERT(this->get_gid());
    typedef hpx::opencl::server::buffer::size_action func;

    return hpx::async<func>(this->get_gid());

}

hpx::lcos::future<hpx::opencl::event>
buffer::enqueue_read(size_t offset, size_t size) const
{
    std::vector<hpx::opencl::event> events(0);
    return enqueue_read(offset, size, events);
}

hpx::lcos::future<hpx::opencl::event>
buffer::enqueue_read(size_t offset, size_t size,
                            hpx::opencl::event event) const
{
    std::vector<hpx::opencl::event> events;
    events.push_back(event);
    return enqueue_read(offset, size, events);
}

hpx::lcos::future<hpx::opencl::event>
buffer::enqueue_read(size_t offset, size_t size,
                            std::vector<hpx::opencl::event> events) const
{

    BOOST_ASSERT(this->get_gid());
    typedef hpx::opencl::server::buffer::read_action func;

    return hpx::async<func>(this->get_gid(), offset, size, events);
}


hpx::lcos::future<hpx::opencl::event>
buffer::enqueue_write(size_t offset, size_t size, const void* data) const
{
    std::vector<hpx::opencl::event> events(0);
    return enqueue_write(offset, size, data, events);
}

hpx::lcos::future<hpx::opencl::event>
buffer::enqueue_write(size_t offset, size_t size, const void* data,
                            hpx::opencl::event event) const
{
    std::vector<hpx::opencl::event> events(1);
    events[0] = event;
    return enqueue_write(offset, size, data, events);
}

hpx::lcos::future<hpx::opencl::event>
buffer::enqueue_write(size_t offset, size_t size, const void* data,
                             std::vector<hpx::opencl::event> events) const
{

    BOOST_ASSERT(this->get_gid());

    // Make data pointer serializable
    hpx::util::serialize_buffer<char>
    serializable_data((char*)const_cast<void*>(data), size,
            hpx::util::serialize_buffer<char>::init_mode::reference);

    // Run write_action
    typedef hpx::opencl::server::buffer::write_action func;

    return hpx::async<func>(this->get_gid(), offset, serializable_data,
                            events);

}


hpx::lcos::future<hpx::opencl::event>
buffer::enqueue_fill(const void* pattern, size_t pattern_size, size_t offset,
                     size_t size) const
{
    std::vector<hpx::opencl::event> events(0);
    return enqueue_fill(pattern, pattern_size, offset, size, events);
}

hpx::lcos::future<hpx::opencl::event>
buffer::enqueue_fill(const void* pattern, size_t pattern_size, size_t offset,
                     size_t size, hpx::opencl::event event) const
{
    std::vector<hpx::opencl::event> events(1);
    events[0] = event;
    return enqueue_fill(pattern, pattern_size, offset, size, events);
}

hpx::lcos::future<hpx::opencl::event>
buffer::enqueue_fill(const void* pattern, size_t pattern_size, size_t offset,
                     size_t size, std::vector<hpx::opencl::event> events) const
{

    BOOST_ASSERT(this->get_gid());

    // Make data pointer serializable
    hpx::util::serialize_buffer<char>
    serializable_pattern((char*)const_cast<void*>(pattern), pattern_size,
            hpx::util::serialize_buffer<char>::init_mode::reference);

    // Run fill_action
    typedef hpx::opencl::server::buffer::fill_action func;
    return hpx::async<func>(this->get_gid(), serializable_pattern, offset, size,
                            events);

}

