// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "clx_event.hpp"





using namespace hpx::opencl;


clx_event::clx_event(hpx::naming::id_type _hosting_device, cl_event _event_ptr)
{
    this->hosting_device = _hosting_device;
    this->cl_event_ptr = (clx_event_id) _event_ptr;
}





hpx::naming::id_type
clx_event::get_device_gid() const
{
    return hosting_device;
}

clx_event_id
clx_event::get_cl_event_id() const
{
    return cl_event_ptr;
}



