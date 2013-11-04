// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/lcos/future.hpp>

#include "server/device.hpp"

#include "device.hpp"
#include "buffer.hpp"


using hpx::opencl::device;

hpx::opencl::buffer
device::clCreateBuffer(cl_mem_flags flags, size_t size)
{

    BOOST_ASSERT(this->get_gid());
    typedef hpx::opencl::server::device::clCreateBuffer_action func;
    
    // Create new Buffer Server
    hpx::lcos::future<hpx::naming::id_type> buffer_server = 
    hpx::async<func>(this->get_gid(), flags, size);
    
    // Return Buffer Client wrapped around Buffer Server
    return buffer(buffer_server);
}

hpx::lcos::future<boost::shared_ptr<std::vector<char>>>
device::get_event_data(hpx::opencl::event event)
{
    BOOST_ASSERT(this->get_gid());

    typedef hpx::opencl::server::device::get_event_data_action func;
    
    return hpx::async<func>(this->get_gid(), event);
}
