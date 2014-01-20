// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "std.hpp"

#include "device.hpp"

hpx::lcos::unique_future<std::vector<hpx::opencl::device>>
hpx::opencl::get_devices( hpx::naming::id_type node_id,
                          cl_device_type device_type,
                          float required_cl_version)
{

    typedef hpx::opencl::server::get_devices_action action;
    return async<action>(node_id, device_type, required_cl_version);

}


