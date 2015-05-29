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

