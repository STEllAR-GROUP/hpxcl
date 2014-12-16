// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "server/device.hpp"

#include "device.hpp"

using hpx::opencl::device;
hpx::future<hpx::util::serialize_buffer<char>>
device::get_device_info(cl_device_info info_type) const
{

    BOOST_ASSERT(this->get_gid());

    typedef hpx::opencl::server::device::get_device_info_action func;

    return hpx::async<func>(this->get_gid(), info_type);

}


hpx::future<hpx::util::serialize_buffer<char>>
device::get_platform_info(cl_platform_info info_type) const
{

    BOOST_ASSERT(this->get_gid());

    typedef hpx::opencl::server::device::get_platform_info_action func;

    return hpx::async<func>(this->get_gid(), info_type);

}

std::string
device::device_info_to_string(hpx::future<hpx::util::serialize_buffer<char>> info)
{

    hpx::util::serialize_buffer<char> char_array = info.get();

    // Calculate length of string. Cut short if it has a 0-Termination
    // (Some queries like CL_DEVICE_NAME always return a size of 64, but 
    // contain a 0-terminated string)
    size_t length = 0;
    while(length < char_array.size())
    {
        if(char_array[length] == '\0') break;
        length++;
    }

    return std::string(char_array.data(), char_array.data() + length);

}
