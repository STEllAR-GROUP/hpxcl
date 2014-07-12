// Copyright (c)        2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_STD_HPP_
#define HPX_OPENCL_SERVER_STD_HPP_

#include <CL/cl.h>

#include <vector>
#include <hpx/hpx.hpp>
#include <hpx/config.hpp>
#include <hpx/include/actions.hpp>

#include <boost/serialization/vector.hpp>

#include "../fwd_declarations.hpp"

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{ namespace server{


    // /////////////////////////////////////////////////////
    //  Global opencl functions
    //  

    // Returns the IDs of all devices on current host
    std::vector<hpx::opencl::device>
    get_devices(cl_device_type, std::string cl_version);

    //[opencl_management_action_types
    HPX_DEFINE_PLAIN_ACTION(get_devices, get_devices_action);
    //]

}}}


HPX_REGISTER_PLAIN_ACTION_DECLARATION(hpx::opencl::server::get_devices_action);


#endif
