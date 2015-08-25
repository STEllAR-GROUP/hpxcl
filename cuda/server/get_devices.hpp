// Copyright (c)        2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CUDA_SERVER_GET_DEVICES_HPP_
#define HPX_CUDA_SERVER_GET_DEVICES_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include <cuda.h>

#include "../fwd_declarations.hpp"

////////////////////////////////////////////////////////////////
namespace hpx { namespace cuda{ namespace server{


    // /////////////////////////////////////////////////////
    //  Global opencl functions
    //

    // Returns the IDs of all devices on current host
    //std::vector<hpx::cuda::device>
    //get_devices(cl_device_type, std::string cl_version);

    //[opencl_management_action_types
    //HPX_DEFINE_PLAIN_ACTION(get_devices, get_devices_action);
    //]

}}}

//HPX_ACTION_USES_LARGE_STACK(hpx::cuda::server::get_devices_action);
//HPX_REGISTER_ACTION_DECLARATION(hpx::cuda::server::get_devices_action,
  //                         hpx_cuda_server_get_devices_action);

#endif
