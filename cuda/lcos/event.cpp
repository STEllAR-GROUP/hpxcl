// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "event.hpp"

#include "../server/device.hpp"


void
hpx::cuda::lcos::detail::unregister_event( hpx::naming::id_type device_id,
                                             hpx::naming::gid_type event_gid )
{

    typedef hpx::cuda::server::device::release_event_action func;
    hpx::apply<func>( device_id, event_gid );

}

//template<>
void hpx::cuda::lcos::detail::event<void, hpx::util::unused_type>::arm(){

    // Tell the device server that we'd like to be informed when the cl_event
    // is completed
    typedef hpx::cuda::server::device::activate_deferred_event_action func;
    hpx::apply<func>(device_id, event_id);

}


