// Copyright (c)        2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_GET_DEVICES_HPP_
#define HPX_OPENCL_GET_DEVICES_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "export_definitions.hpp"

#include <CL/cl.h>

#include "fwd_declarations.hpp"

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{

    /**
     * @brief Fetches a list of accelerator devices present on target node.
     *
     * It is recommended to only use OpenCL Version >= 1.1.
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
     *                            Version number must have the following format:
     *                            "OpenCL <major>.<minor>"<BR>
     *                            Recommended value is "OpenCL 1.1".
     * @return A list of suitable OpenCL devices on target node
     */
    HPX_OPENCL_EXPORT
    hpx::lcos::future<std::vector<device>>
    get_devices( hpx::naming::id_type node_id, cl_device_type device_type,
                 std::string required_cl_version );

    /**
     * @brief Fetches a list of all accelerator devices present in the current 
     *        hpx environment.
     *
     * It is recommended to only use OpenCL Version >= 1.1.
     * Earlier devices seem to be blocking on every enqueue-call, which
     * is counter-productive to the general idea of the hpx framework.
     *
     * @param device_type         The device type, according to OpenCL standard.
     *                            <BR>
     *                            For further information, look at the official 
     *                            <A HREF="http://www.khronos.org/registry/cl/sd
     * k/1.2/docs/man/xhtml/clGetDeviceIDs.html">
     *                            OpenCL Reference</A>.
     * @param required_cl_version All devices that don't support this OpenCL
     *                            version will be ignored.<BR>
     *                            Version number must have the following format:
     *                            "OpenCL <major>.<minor>"<BR>
     *                            Recommended value is "OpenCL 1.1".
     * @return A list of suitable OpenCL devices
     */
    HPX_OPENCL_EXPORT
    hpx::lcos::future<std::vector<device>>
    get_all_devices( cl_device_type device_type,
                     std::string required_cl_version );

}}



#endif

