// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_DEVICE_HPP_
#define HPX_OPENCL_SERVER_DEVICE_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include <CL/cl.h>

#include <hpx/runtime/components/server/managed_component_base.hpp>

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{ namespace server{
    

    // /////////////////////////////////////////////////////
    // This class represents an OpenCL accelerator device.
    //
    class device
      : public hpx::components::managed_component_base<device>
    {
    public:
        // Constructor
        device();
        ~device();

        //////////////////////////////////////////////////
        /// Local public functions
        ///
        void init(cl_device_id device_id, bool enable_profiling=false);


        //////////////////////////////////////////////////
        /// Exposed functionality of this component
        ///

        // returns device specific information
        hpx::util::serialize_buffer<char>
        get_device_info(cl_device_info info_type);
        
        // returns platform specific information
        hpx::util::serialize_buffer<char>
        get_platform_info(cl_platform_info info_type);

    HPX_DEFINE_COMPONENT_ACTION(device, get_device_info);
    HPX_DEFINE_COMPONENT_ACTION(device, get_platform_info);

    private:
        ///////////////////////////////////////////////
        // Private Member Functions
        //
        
        // Error Callback
        static void CL_CALLBACK error_callback(const char*, const void*,
                                               std::size_t, void*);

    private:
        ///////////////////////////////////////////////
        // Private Member Variables
        //
        cl_device_id        device_id;
        cl_platform_id      platform_id;
        cl_context          context;
        cl_command_queue    command_queue;

    };
}}}

//[opencl_management_registration_declarations
HPX_ACTION_USES_LARGE_STACK(hpx::opencl::server::device::get_device_info_action);
HPX_REGISTER_ACTION_DECLARATION(
        hpx::opencl::server::device::get_device_info_action,
        opencl_device_get_device_info_action);
HPX_ACTION_USES_LARGE_STACK(hpx::opencl::server::device::get_platform_info_action);
HPX_REGISTER_ACTION_DECLARATION(
        hpx::opencl::server::device::get_platform_info_action,
        opencl_device_get_platform_info_action);
//]

#endif
