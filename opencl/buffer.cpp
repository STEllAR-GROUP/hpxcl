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

#include "enqueue_overloads.hpp"

using hpx::opencl::buffer;



hpx::lcos::unique_future<size_t>
buffer::size() const
{
    
    BOOST_ASSERT(this->get_gid());
    typedef hpx::opencl::server::buffer::size_action func;

    return hpx::async<func>(this->get_gid());

}




// ////////////////////////////////
// OVERLOAD DEFINITIONS
//

HPX_OPENCL_OVERLOAD_FUNCTION(buffer, enqueue_read, 
                             size_t offset COMMA size_t size,
                             offset COMMA size);

HPX_OPENCL_OVERLOAD_FUNCTION(buffer, enqueue_write,
                         size_t offset COMMA size_t size COMMA const void* data,
                         offset COMMA size COMMA data);

#ifdef CL_VERSION_1_2
HPX_OPENCL_OVERLOAD_FUNCTION(buffer, enqueue_fill,
                            const void* pattern COMMA size_t pattern_size COMMA
                            size_t offset COMMA size_t size,
                            pattern COMMA pattern_size COMMA offset COMMA size);
#endif

HPX_OPENCL_OVERLOAD_FUNCTION(buffer, enqueue_copy,
                             buffer src COMMA size_t src_offset COMMA
                             size_t dst_offset COMMA size_t size,
                             src COMMA src_offset COMMA dst_offset COMMA size);




// ///////////////////////////////////////////////////////
//  FUNCTION DEFINITIONS
//

hpx::lcos::unique_future<hpx::opencl::event>
buffer::enqueue_read(size_t offset, size_t size,
                            std::vector<hpx::opencl::event> events) const
{

    BOOST_ASSERT(this->get_gid());
    typedef hpx::opencl::server::buffer::read_action func;

    return hpx::async<func>(this->get_gid(), offset, size, events);
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

#ifdef CL_VERSION_1_2
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
#endif


// Copy Buffer
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


