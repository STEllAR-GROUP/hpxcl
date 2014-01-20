// Copyright (c)        2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_STD_HPP__
#define HPX_OPENCL_STD_HPP__

#include "server/std.hpp"

#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>

#include <CL/cl.h>

#include <vector>

#include "fwd_declarations.hpp"

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{

    /**
     * @brief Fetches a list of accelerator devices present on target node.
     *
     * It is recommended to only use OpenCL Version >= 1.1f.
     * Earlier devices seem to be blocking on every enqueue-call, which
     * is counter-productive to the general idea of the hpx framework.
     *
     * @param node_id             The ID of the target node
     * @param device_type         The device type, according to OpenCL standard.
     *                            <BR>
     *                            For further information, look at the official 
     *                            <A HREF="http://www.khronos.org/registry/cl/sd
     * k/1.2/docs/man/xhtml/clGetDeviceIDs.html">
     *                            OpenCL Reference</A>.
     * @param required_cl_version All devices that don't support this OpenCL
     *                            version will be ignored.<BR>
     *                            Recommended value is 1.1f.
     * @return A list of suitable OpenCL devices on target node
     */
    hpx::lcos::unique_future<std::vector<device>>
    get_devices( hpx::naming::id_type node_id, cl_device_type device_type,
                 float required_cl_version );

}}



#endif
