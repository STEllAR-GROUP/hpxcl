// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "server/device.hpp"

#include "device.hpp"
#include "buffer.hpp"
#include "program.hpp"
#include "event.hpp"

#include <boost/serialization/vector.hpp>

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/serialize_buffer.hpp>
#include <hpx/lcos/future.hpp>




using hpx::opencl::device;

hpx::opencl::buffer
device::create_buffer(cl_mem_flags flags, size_t size, const void* data) const
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
    return buffer(std::move(buffer_server));

}

hpx::opencl::buffer
device::create_buffer(cl_mem_flags flags, size_t size) const
{

    BOOST_ASSERT(this->get_gid());
    
    // Create new Buffer Server
    hpx::lcos::future<hpx::naming::id_type>
    buffer_server = hpx::components::new_<hpx::opencl::server::buffer>
                    (get_colocation_id_sync(get_gid()), get_gid(), flags, size);

    // Return Buffer Client wrapped around Buffer Server
    return buffer(std::move(buffer_server));

}

hpx::opencl::program
device::create_program_with_source(std::string source) const
{

    BOOST_ASSERT(this->get_gid());

    // Create new program object server
    hpx::lcos::future<hpx::naming::id_type>
    program_server = hpx::components::new_<hpx::opencl::server::program>
                     (get_colocation_id_sync(get_gid()), get_gid(), source);

    // Return program object client
    return program(std::move(program_server));

}

hpx::opencl::program
device::create_program_with_binary(size_t binary_size, const char* binary) const
{

    BOOST_ASSERT(this->get_gid());

    // Make data pointer serializable
    hpx::util::serialize_buffer<char>
    serializable_binary(const_cast<char*>(binary), binary_size,
                        hpx::util::serialize_buffer<char>::init_mode::reference);

    // Create new program object server
    hpx::lcos::future<hpx::naming::id_type>
    program_server = hpx::components::new_<hpx::opencl::server::program>
                     (get_colocation_id_sync(get_gid()), get_gid(),
                                                           serializable_binary);

    // Return program object client
    return program(std::move(program_server));

}

hpx::lcos::future<hpx::opencl::event>
device::create_user_event() const
{
    BOOST_ASSERT(this->get_gid());

    typedef hpx::opencl::server::device::create_user_event_action func;

    return hpx::async<func>(this->get_gid());
}

hpx::lcos::future<std::vector<char>>
device::get_device_info(cl_device_info info_type) const
{

    BOOST_ASSERT(this->get_gid());

    typedef hpx::opencl::server::device::get_device_info_action func;

    return hpx::async<func>(this->get_gid(), info_type);

}

hpx::lcos::future<std::vector<char>>
device::get_platform_info(cl_platform_info info_type) const
{

    BOOST_ASSERT(this->get_gid());

    typedef hpx::opencl::server::device::get_platform_info_action func;

    return hpx::async<func>(this->get_gid(), info_type);

}

std::string
device::device_info_to_string(hpx::lcos::future<std::vector<char>> info)
{

    std::vector<char> char_array = info.get();

    // Calculate length of string. Cut short if it has a 0-Termination
    // (Some queries like CL_DEVICE_NAME always return a size of 64, but 
    // contain a 0-terminated string)
    size_t length = 0;
    while(length < char_array.size())
    {
        if(char_array[length] == '\0') break;
        length++;
    }

    return std::string(char_array.begin(), char_array.begin() + length);

}

// used for create_future_event_externally, this is the event_future callback
static void
trigger_user_event_activator_callback(
                      hpx::lcos::shared_future<hpx::opencl::event> event_future)
{

    // get the event
    hpx::opencl::event event = event_future.get();

    // trigger the event
    event.trigger();

}

// used for create_future_event, this is the future.then callback
void
device::trigger_user_event_externally(
                      hpx::lcos::shared_future<hpx::opencl::event> event_future)
{

    event_future.then(hpx::util::bind(trigger_user_event_activator_callback,
                                      util::placeholders::_1));
                                      
}


