// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/serialize_buffer.hpp>
#include <hpx/lcos/future.hpp>

#include "server/device.hpp"

#include "device.hpp"
#include "buffer.hpp"


using hpx::opencl::device;

hpx::opencl::buffer
device::create_buffer(cl_mem_flags flags, size_t size, const void* data)
{
   
    BOOST_ASSERT(this->get_gid());

    // Make data pointer serializable
    hpx::util::serialize_buffer<char>
    serializable_data((char*)const_cast<void*>(data), size,
          hpx::util::serialize_buffer<char>::init_mode::reference);

    // Create new Buffer Server
    hpx::lcos::future<hpx::naming::id_type>
    buffer_server = hpx::components::new_<hpx::opencl::server::buffer>
                    (get_colocation_id_sync(get_gid()), get_gid(), flags, size,
                     serializable_data);
    
    // Return Buffer Client wrapped around Buffer Server
    return buffer(buffer_server);

}

hpx::opencl::buffer
device::create_buffer(cl_mem_flags flags, size_t size)
{

    BOOST_ASSERT(this->get_gid());
    
    // Create new Buffer Server
    hpx::lcos::future<hpx::naming::id_type>
    buffer_server = hpx::components::new_<hpx::opencl::server::buffer>
                    (get_colocation_id_sync(get_gid()), get_gid(), flags, size);

    // Return Buffer Client wrapped around Buffer Server
    return buffer(buffer_server);

}

hpx::opencl::program
device::create_program_with_source(std::string source)
{

    BOOST_ASSERT(this->get_gid());

    // Create new program object server
    hpx::lcos::future<hpx::naming::id_type>
    program_server = hpx::components::new_<hpx::opencl::server::program>
                     (get_colocation_id_sync(get_gid()), get_gid(), source);

    // Return program object client
    return program(program_server);

}

hpx::lcos::future<hpx::opencl::event>
device::create_user_event()
{
    BOOST_ASSERT(this->get_gid());

    typedef hpx::opencl::server::device::create_user_event_action func;

    return hpx::async<func>(this->get_gid());
}

// used for create_future_event, this is the future.then callback
void
device::trigger_user_event_externally(hpx::lcos::future<hpx::opencl::event> event)
{
    event.get().trigger();
}


