// Copyright (c)        2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_GET_DEVICES_HPP_
#define HPX_OPENCL_SERVER_GET_DEVICES_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "../cl_headers.hpp"

#include "../fwd_declarations.hpp"
#include "../device.hpp"

////////////////////////////////////////////////////////////////
namespace hpx {
namespace opencl {
namespace server {

// /////////////////////////////////////////////////////
//  Global opencl functions
//

// Returns the IDs of all devices on current host
std::vector<hpx::opencl::device> create_devices(cl_device_type,
                                                std::string cl_version);

//[opencl_management_action_types
HPX_DEFINE_PLAIN_ACTION(create_devices, create_devices_action);
//]

}  // namespace server
}  // namespace opencl
}  // namespace hpx

HPX_ACTION_USES_MEDIUM_STACK(hpx::opencl::server::create_devices_action);
HPX_REGISTER_ACTION_DECLARATION(hpx::opencl::server::create_devices_action,
                                hpx_opencl_server_create_devices_action)

#endif
