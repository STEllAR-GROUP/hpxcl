// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_DEVICE_HPP_
#define HPX_OPENCL_DEVICE_HPP_


#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "export_definitions.hpp"
#include "server/device.hpp"

#include <hpx/include/components.hpp>
#include <hpx/lcos/future.hpp>

#include "fwd_declarations.hpp"

namespace hpx {
namespace opencl {

    /////////////////////////////////////////
    /// @brief An accelerator device.
    ///
    class HPX_OPENCL_EXPORT device
      : public hpx::components::client_base<
          device, hpx::components::stub_base<server::device>
        >
    
    {
    
        typedef hpx::components::client_base<
            device, hpx::components::stub_base<server::device>
            > base_type;

        public:
            device(){}

            device(hpx::shared_future<hpx::naming::id_type> const& gid)
              : base_type(gid)
            {}
            
            // ///////////////////////////////////////
            // Exposed Component functionality
            // 

            /**
             *  @brief Queries device infos.
             *  
             *  @param info_type    The type of information.<BR>
             *                      A complete list can be found on the official
             *                      <A HREF="http://www.khronos.org/registry/cl/
             * sdk/1.2/docs/man/xhtml/clGetDeviceInfo.html">
             *                      OpenCL Reference</A>.
             *  @return The info data as char array.<BR>
             *          This will typically be cast to some other type via
             *          (for example):
             *          \code{.cpp}
             *          cl_uint *return_uint = (cl_uint*)return_charvector.data();
             *          \endcode
             *          or converted to a string via \ref device_info_to_string.
             */
            hpx::future<hpx::util::serialize_buffer<char>>
            get_device_info(cl_device_info info_type) const;

             /**
             *  @brief Queries platform infos.
             *  
             *  @param info_type    The type of information.<BR>
             *                      A complete list can be found on the official
             *                      <A HREF="http://www.khronos.org/registry/cl/
             * sdk/1.2/docs/man/xhtml/clGetPlatformInfo.html">
             *                      OpenCL Reference</A>.
             *  @return The info data as char array.<BR>
             *          This will typically be cast to some other type via
             *          (for example):
             *          \code{.cpp}
             *          cl_uint *return_uint = (cl_uint*)return_charvector.data();
             *          \endcode
             *          or converted to a string via \ref device_info_to_string.
             */
            hpx::future<hpx::util::serialize_buffer<char>>
            get_platform_info(cl_platform_info info_type) const;

            /** 
             *  @brief Converts device info data to a string
             *
             *  This method is for convenience.<BR>
             *  It should only be used on String return types of
             *  \ref get_device_info.
             *
             *  @param info     Output data of \ref get_device_info. <BR>
             *                  Only use this function if the data type is
             *                  a string.
             *  @return         The data, converted to a string.
             */
            static std::string
            device_info_to_string(hpx::lcos::future<
                                       hpx::util::serialize_buffer<char>> info);

    };

}}


#endif// HPX_OPENCL_DEVICE_HPP_

            
