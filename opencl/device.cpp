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


HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<
                        hpx::opencl::server::device> device_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(device_type, device);

HPX_REGISTER_ACTION(device_type::wrapped_type::test_action, device_test_action);
HPX_REGISTER_ACTION(device_type::wrapped_type::clCreateBuffer_action,
                    device_clCreateBuffer_action);



hpx::opencl::buffer
hpx::opencl::device::clCreateBuffer(cl_mem_flags flags, size_t size)
{

    BOOST_ASSERT(this->get_gid());
    typedef hpx::opencl::server::device::clCreateBuffer_action create_buffer_func;
    
    // Create new Buffer Server
    hpx::lcos::future<hpx::naming::id_type> buffer_server = 
    hpx::async<create_buffer_func>(this->get_gid(), flags, size);
    
    // Return Buffer Client wrapped around Buffer Server
    return buffer(buffer_server);
}

