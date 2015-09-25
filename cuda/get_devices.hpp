// Copyright (c)        2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CUDA_GET_DEVICES_HPP_
#define HPX_CUDA_GET_DEVICES_HPP_

#include <hpx/include/lcos.hpp>

#include "cuda/fwd_declarations.hpp"
#include "cuda/export_definitions.hpp"
#include "cuda/server/get_devices.hpp"
#include "cuda.hpp"

#include <cuda.h>

#include <vector>

namespace hpx {
namespace cuda {

/**
 * \brief Fetches a list of accelerator devices present on target node.
 *
 * \param node_id The ID of the target node
 * \param major The minimal major version of the CUDA device
 * \param minor The minimal minor version of the CUDA device
 *
 * \return A list of suitable CUDA devices on target node
 */
HPX_CUDA_EXPORT hpx::future<std::vector<device> >
get_devices(hpx::naming::id_type node_id, int major = 1, int minor = 0);

/**
 * \brief Fetches a list of all accelerator devices present in the current
 *        hpx environment.
 *
 * \param major The minimal major version of the CUDA device
 * \param minor The minimal minor version of the CUDA device
 *
 * \return A list of suitable CUDA devices
 */
HPX_CUDA_EXPORT hpx::future<std::vector<device>>
get_all_devices(int major = 1, int minor = 0);

/**
 * \brief Fetches a list of local accelerator devices present in the current
 *        hpx environment.
 *
 * \param major The minimal major version of the CUDA device
 * \param minor The minimal minor version of the CUDA device
 *
 * \return A list of suitable CUDA devices
 */
HPX_CUDA_EXPORT hpx::future<std::vector<device>>
get_local_devices(int major = 1, int minor = 0);

/**
 * \brief Fetches a list of remote accelerator devices present in the current
 *        hpx environment.
 *
 * \param major The minimal major version of the CUDA device
 * \param minor The minimal minor version of the CUDA device
 *
 * \return A list of suitable CUDA devices
 */
HPX_CUDA_EXPORT hpx::future<std::vector<device>>
get_remote_devices(int major = 1, int minor = 0);
}
}

#endif
