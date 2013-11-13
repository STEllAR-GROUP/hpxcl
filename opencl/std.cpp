// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "std.hpp"

using hpx::opencl::clx_device_id;

hpx::lcos::future<std::vector<clx_device_id>>
hpx::opencl::get_device_ids(hpx::naming::id_type node_id,
                            cl_device_type device_type,
                            float required_cl_version)
{

    typedef hpx::opencl::server::get_device_ids_action action;
    return async<action>(node_id, device_type, required_cl_version);

}


hpx::lcos::future<std::vector<char>>
hpx::opencl::get_device_info( hpx::naming::id_type  node_id,
                              clx_device_id         device_id,
                              cl_device_info        info_type )
{
    
    // Retrieve info from node
    typedef hpx::opencl::server::get_device_info_action action;
    return async<action>(node_id, device_id, info_type);
    
}











