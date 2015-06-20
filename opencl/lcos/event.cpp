// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "event.hpp"

#include "../server/device.hpp"

void
hpx::opencl::lcos::detail::unregister_event( hpx::naming::id_type device_id,
                                             hpx::naming::gid_type event_gid )
{

    typedef hpx::opencl::server::device::release_event_action func;
    hpx::apply<func>( device_id, event_gid );

}

//template<>
void hpx::opencl::lcos::detail::event<void, hpx::util::unused_type>::arm(){

    // Tell the device server that we'd like to be informed when the cl_event
    // is completed
    typedef hpx::opencl::server::device::activate_deferred_event_action func;
    hpx::apply<func>(device_id, this->get_event_id());

}


