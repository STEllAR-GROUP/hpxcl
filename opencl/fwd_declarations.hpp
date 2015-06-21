// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_FWD_DECLARATIONS_HPP_
#define HPX_OPENCL_FWD_DECLARATIONS_HPP_


// This file forward-declares all hpxcl classes.
// This is important to remove circular dependencies and improve compile speed.

#include "export_definitions.hpp"

namespace hpx {

/// The OpenCL client namespace
namespace opencl {

    class device;
    class buffer;
    class program;
    class kernel;

    // The OpenCL server namespace
    namespace server {

        class HPX_OPENCL_EXPORT device;
        class HPX_OPENCL_EXPORT buffer;
        class HPX_OPENCL_EXPORT program;
        class HPX_OPENCL_EXPORT kernel;

    }

}}







#endif


