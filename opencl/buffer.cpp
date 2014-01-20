// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "buffer.hpp"

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include "event.hpp"

using hpx::opencl::buffer;


hpx::lcos::unique_future<size_t>
buffer::size() const
{
    
    BOOST_ASSERT(this->get_gid());
    typedef hpx::opencl::server::buffer::size_action func;

    return hpx::async<func>(this->get_gid());

}

hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_read(size_t offset, size_t size) const
{
    std::vector<hpx::opencl::event> events(0);
    return enqueue_read(offset, size, events);
}

hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_read(size_t offset, size_t size,
                            hpx::opencl::event event) const
{
    std::vector<hpx::opencl::event> events;
    events.push_back(event);
    return enqueue_read(offset, size, events);
}

hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_read(size_t offset, size_t size,
                            std::vector<hpx::opencl::event> events) const
{

    BOOST_ASSERT(this->get_gid());
    typedef hpx::opencl::server::buffer::read_action func;

    return hpx::async<func>(this->get_gid(), offset, size, events);
}

hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_read(size_t offset, size_t size,
                     hpx::lcos::shared_future<hpx::opencl::event> event) const
{
    // Create event list
    std::vector<hpx::lcos::shared_future<hpx::opencl::event>> events;

    // Add the event to event list
    events.push_back(std::move(event));
    
    // Proxy to enqueue_read for event lists
    return enqueue_read(offset, size, std::move(events));
}

hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_read(size_t offset, size_t size,
               std::vector<hpx::lcos::shared_future<hpx::opencl::event>> events) const
{
/*    BOOST_ASSERT(this->get_gid());

    // define the async call
    future_call_def_2(buffer, size_t, size_t, enqueue_read); 

    // run the async call
    return future_call::run(*this, offset, size, events);

*/
    return unique_future<hpx::opencl::event>();
}



hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_write(size_t offset, size_t size, const void* data) const
{
    std::vector<hpx::opencl::event> events(0);
    return enqueue_write(offset, size, data, events);
}

hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_write(size_t offset, size_t size, const void* data,
                            hpx::opencl::event event) const
{
    std::vector<hpx::opencl::event> events(1);
    events[0] = event;
    return enqueue_write(offset, size, data, events);
}


hpx::lcos::unique_future<hpx::opencl::event>
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

hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_write(size_t offset, size_t size, const void* data,
                     hpx::lcos::shared_future<hpx::opencl::event> event) const
{
    std::vector<hpx::lcos::shared_future<hpx::opencl::event>> events;
    events.push_back(std::move(event));
    return enqueue_write(offset, size, data, std::move(events));
}

hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_write(size_t offset, size_t size, const void* data,
               std::vector<hpx::lcos::shared_future<hpx::opencl::event>> events) const
{
/*
 * BOOST_ASSERT(this->get_gid());

    // define the async call
    future_call_def_3(buffer, size_t, size_t, const void*, enqueue_write); 

    // run the async call
    return future_call::run(*this, offset, size, data, events);

*/
    return unique_future<hpx::opencl::event>();
}



hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_fill(const void* pattern, size_t pattern_size, size_t offset,
                     size_t size) const
{
    std::vector<hpx::opencl::event> events(0);
    return enqueue_fill(pattern, pattern_size, offset, size, events);
}

hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_fill(const void* pattern, size_t pattern_size, size_t offset,
                     size_t size, hpx::opencl::event event) const
{
    std::vector<hpx::opencl::event> events(1);
    events[0] = event;
    return enqueue_fill(pattern, pattern_size, offset, size, events);
}

hpx::lcos::unique_future<hpx::opencl::event>
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

hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_fill(const void* pattern, size_t pattern_size, size_t offset,
                     size_t size,
                     hpx::lcos::shared_future<hpx::opencl::event> event) const
{
    // Create events list
    std::vector<hpx::lcos::shared_future<hpx::opencl::event>> events;
    
    // Add the event to the list
    events.push_back(std::move(event));
    
    // Proxy to enqueue_fill for event lists
    return enqueue_fill(pattern, pattern_size, offset, size, std::move(events));
}

hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_fill(const void* pattern, size_t pattern_size, size_t offset,
                     size_t size, 
               std::vector<hpx::lcos::shared_future<hpx::opencl::event>> events) const
{
/*    BOOST_ASSERT(this->get_gid());

    // define the async call
    future_call_def_4(buffer, const void*, size_t, size_t, size_t, enqueue_fill); 

    // run the async call
    return future_call::run(*this, pattern, pattern_size, offset, size, events);

*/
    return unique_future<hpx::opencl::event>();
}


// Copy Buffer
hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_copy(buffer src, size_t src_offset, size_t dst_offset,
                                 size_t size) const
{
 
     std::vector<hpx::opencl::event> events(0);
     return enqueue_copy(src, src_offset, dst_offset, size, events);    
    
}

hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_copy(buffer src, size_t src_offset, size_t dst_offset,
                                 size_t size,
                                 hpx::opencl::event event) const
{
 
     std::vector<hpx::opencl::event> events(1);
     events[0] = event;
     return enqueue_copy(src, src_offset, dst_offset, size, events);    
    
}



hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_copy(buffer src, size_t src_offset, size_t dst_offset,
                     size_t size,
                     std::vector<hpx::opencl::event> events) const
{
    BOOST_ASSERT(this->get_gid());
    BOOST_ASSERT(src.get_gid());

    // Create dimensions vector
    std::vector<size_t> dim(3);
    dim[0] = src_offset;
    dim[1] = dst_offset;
    dim[2] = size;

    // Run copy_action
    typedef hpx::opencl::server::buffer::copy_action func;
    return hpx::async<func>(this->get_gid(), src.get_gid(), dim, events);

}

hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_copy(buffer src, size_t src_offset, size_t dst_offset,
                     size_t size,
                     hpx::lcos::shared_future<hpx::opencl::event> event) const
{
    std::vector<hpx::lcos::shared_future<hpx::opencl::event>> events;
    events.push_back(std::move(event));
    return enqueue_copy(src, src_offset, dst_offset, size, std::move(events));
}

hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_copy(buffer src, size_t src_offset, size_t dst_offset,
                     size_t size,
               std::vector<hpx::lcos::shared_future<hpx::opencl::event>> events) const
{
/*
 * BOOST_ASSERT(this->get_gid());

    // define the async call
    future_call_def_4(buffer, buffer, size_t, size_t, size_t, enqueue_copy); 

    // run the async call
    return future_call::run(*this, src, src_offset, dst_offset, size, events);

*/
    return unique_future<hpx::opencl::event>();
}


