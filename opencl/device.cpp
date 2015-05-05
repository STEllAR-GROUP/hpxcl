// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The class header file
#include "device.hpp"

// The server class
#include "server/device.hpp"

// Dependencies
#include "buffer.hpp"

using hpx::opencl::device;
hpx::future<hpx::serialization::serialize_buffer<char>>
device::get_device_info(cl_device_info info_type) const
{

    HPX_ASSERT(this->get_gid());

    typedef hpx::opencl::server::device::get_device_info_action func;

    return hpx::async<func>(this->get_gid(), info_type);

}


hpx::future<hpx::serialization::serialize_buffer<char>>
device::get_platform_info(cl_platform_info info_type) const
{

    HPX_ASSERT(this->get_gid());

    typedef hpx::opencl::server::device::get_platform_info_action func;

    return hpx::async<func>(this->get_gid(), info_type);

}

std::string
device::device_info_to_string(
    hpx::future<hpx::serialization::serialize_buffer<char>> info)
{

    hpx::serialization::serialize_buffer<char> char_array = info.get();

    // Calculate length of string. Cut short if it has a 0-Termination
    // (Some queries like CL_DEVICE_NAME always return a size of 64, but 
    // contain a 0-terminated string)
    std::size_t length = 0;
    while(length < char_array.size())
    {
        if(char_array[length] == '\0') break;
        length++;
    }

    return std::string(char_array.data(), char_array.data() + length);

}

hpx::opencl::buffer
device::create_buffer(cl_mem_flags flags, std::size_t size) const
{

    HPX_ASSERT(this->get_gid());

    typedef hpx::opencl::server::device::create_buffer_action func;
    
    hpx::future<hpx::id_type> buffer_server =
                                 hpx::async<func>(this->get_gid(), flags, size);

    return buffer(std::move(buffer_server));

}
