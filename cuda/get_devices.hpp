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


////////////////////////////////////////////////////////////////
namespace hpx {
namespace cuda {

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
HPX_CUDA_EXPORT hpx::future<std::vector<device> >
get_devices( hpx::naming::id_type node_id,
        int major = 1 , int minor = 0 );
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
HPX_CUDA_EXPORT hpx::future<std::vector<device>>
get_all_devices( int major = 1, int minor = 0 );
/**
 * @brief Fetches a list of local accelerator devices present in the current
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
HPX_CUDA_EXPORT hpx::future<std::vector<device>>
get_local_devices(int major = 1, int minor = 0);
/**
 * @brief Fetches a list of remote accelerator devices present in the current
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
HPX_CUDA_EXPORT hpx::future<std::vector<device>>
get_remote_devices(
        int major = 1, int minor = 0 );
}
}

#endif
