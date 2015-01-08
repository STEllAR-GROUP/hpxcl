// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#ifndef HPX_CUDA_STD_HPP_
#define HPX_CUDA_STD_HPP_

#include <hpx/config.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>

#include <cuda.h>
#include <vector>
#include "fwd_declarations.hpp"

namespace hpx
{
	namespace cuda
	{
		hpx::lcos::future<std::vector<device>>
		get_devices(hpx::naming::id_type node_id);
	}
}
#endif //HPX_CUDA_STD_HPP_