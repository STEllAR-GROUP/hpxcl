// Copyright (c)        2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CUDA_SERVER_GET_DEVICES_HPP_
#define HPX_CUDA_SERVER_GET_DEVICES_HPP_

#include <hpx/include/components.hpp>

#include "cuda/fwd_declarations.hpp"
#include "cuda.hpp"
#include "cuda/cuda_error_handling.hpp"
#include "cuda/export_definitions.hpp"

#include <cuda.h>

#include <vector>

////////////////////////////////////////////////////////////////
namespace hpx {
namespace cuda {
namespace server {

// /////////////////////////////////////////////////////
//  Global cuda functions

// Returns the IDs of all devices on current host
HPX_CUDA_EXPORT std::vector<hpx::cuda::device> get_devices(int major, int minor);

HPX_DEFINE_PLAIN_ACTION(get_devices, get_devices_action);

}
}
}

#endif
