// Copyright (c)        2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_STD_HPP__
#define HPX_OPENCL_STD_HPP__

#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>

#include <CL/cl.h>

#include <vector>

#include "server/std.hpp"
#include "name_definitions.hpp"

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{

    // Get all devices on node.
    //      It is recommended to only use OpenCL Version >= 1.1f.
    //      Earlier devices seem to be blocking on every enqueue-call, which
    //      seems counter-productive to the general idea of the hpx framework.
    hpx::lcos::future<std::vector<clx_device_id>>
    get_device_ids( hpx::naming::id_type node_id, cl_device_type device_type,
                    float required_cl_version);

    // Get device information
    hpx::lcos::future<std::vector<char>>
    get_device_info( hpx::naming::id_type          node_id,
                     clx_device_id                 device_id,
                     cl_device_info                info_type);

}}



#endif
