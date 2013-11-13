// Copyright (c)        2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_STD_HPP__
#define HPX_OPENCL_SERVER_STD_HPP__

#include <CL/cl.h>

#include <vector>
#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>

#include <boost/serialization/vector.hpp>

#include "../name_definitions.hpp"

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{ namespace server{

    ////////////////////////////////////////////////////////
    /// Global opencl functions
    /// 

    // Returns the IDs of all devices on current host
    std::vector<clx_device_id> get_device_ids(cl_device_type, float cl_version);
    // Returns informations about given device
    std::vector<char> get_device_info(clx_device_id, cl_device_info);

    //[opencl_management_action_types
    HPX_DEFINE_PLAIN_ACTION(get_device_ids, get_device_ids_action);
    HPX_DEFINE_PLAIN_ACTION(get_device_info, get_device_info_action);
    //]

}}}


HPX_REGISTER_PLAIN_ACTION_DECLARATION(hpx::opencl::server::get_device_ids_action);
HPX_REGISTER_PLAIN_ACTION_DECLARATION(hpx::opencl::server::get_device_info_action);


#endif
