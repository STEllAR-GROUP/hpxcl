// Copyright (c)    2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#pragma once
#ifndef HPX_CUDA_FWD_DECLARATIONS_HPP_
#define HPX_CUDA_FWD_DECLARATIONS_HPP_

#include "export_definitions.hpp"

namespace hpx {
namespace cuda {
class device;
class buffer;
class program;

namespace server {
class HPX_CUDA_EXPORT device;
class HPX_CUDA_EXPORT buffer;
class HPX_CUDA_EXPORT program;
}  // namespace server
}  // namespace cuda
}  // namespace hpx

#endif
