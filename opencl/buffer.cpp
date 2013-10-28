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

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<
                        hpx::opencl::server::buffer> buffer_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(buffer_type, buffer);
HPX_REGISTER_ACTION(buffer_type::wrapped_type::clEnqueueReadBuffer_action,
                    buffer_clEnqueueReadBuffer_action);


hpx::lcos::future<hpx::opencl::event>
buffer::clEnqueueReadBuffer(size_t offset, size_t size,
                            std::vector<hpx::opencl::event> events)
{

    BOOST_ASSERT(this->get_gid());
    typedef hpx::opencl::server::buffer::clEnqueueReadBuffer_action func;

    return hpx::async<func>(this->get_gid(), offset, size, events);
}

