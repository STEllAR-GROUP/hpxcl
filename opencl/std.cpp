// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "std.hpp"

#include <algorithm>

using hpx::opencl::clx_device_id;


std::vector<clx_device_id>
hpx::opencl::get_device_ids(hpx::naming::id_type node_id,
                            cl_device_type device_type)
{
    typedef hpx::opencl::server::clGetDeviceIDs_action action;
    return async<action>(node_id, device_type).get();
}


void
hpx::opencl::get_device_info( hpx::naming::id_type  node_id,
                              clx_device_id         device_id,
                              cl_device_info        info_type,
                              size_t                param_value_size,
                              void*                 param_value,
                              size_t*               param_value_size_ret)    
{
    // Retrieve info from node
    typedef hpx::opencl::server::clGetDeviceInfo_action action;
    std::vector<char> info = async<action>(node_id, device_id, info_type).get();
    
    // Write info to param_value
    if(param_value_size > 0)
    {
        if(param_value == NULL)
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                                "hpx::opencl::clGetDeviceInfo",
                                "param_value must not be NULL");
        memcpy(param_value, &info[0], std::min(info.size(), param_value_size));
    }

    // Write info size to param_value_size_ret
    if(param_value_size_ret != NULL)
    {
        *param_value_size_ret = info.size();
    }

}
