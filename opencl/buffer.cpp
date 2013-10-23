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
HPX_REGISTER_ACTION(buffer_type::wrapped_type::clEnqueueReadBuffer2_action,
                    buffer_clEnqueueReadBuffer2_action);


hpx::lcos::future<hpx::opencl::clx_event>
buffer::clEnqueueReadBuffer(size_t offset, size_t size, bool ptr,
                            std::vector<clx_event> events_)
{

    BOOST_ASSERT(this->get_gid());
    typedef hpx::opencl::server::buffer::clEnqueueReadBuffer2_action func;

    // Convert clx_event list to clx_event_id list
    std::vector<clx_event_id> events(events_.size());
    BOOST_FOREACH(const clx_event & event, events_)
    {
        events.push_back(event.get_cl_event_id());
    }

    return hpx::async<func>(this->get_gid(), offset, size, ptr, events);
}

