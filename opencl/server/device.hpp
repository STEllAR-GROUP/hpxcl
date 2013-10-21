// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_DEVICE_HPP__
#define HPX_OPENCL_SERVER_DEVICE_HPP__

#include <cstdint>

#include <hpx/include/iostreams.hpp>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>

#include <CL/cl.h>

#include "../std.hpp"

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{ namespace server{
    
    ////////////////////////////////////////////////////////
    /// This class represents an OpenCL accelerator device.
    ///
    class device
      : public hpx::components::locking_hook<
          hpx::components::managed_component_base<device>
        >
    {
    public:
        // Constructor
        device();
        device(clx_device_id device_id, bool enable_profiling=false);

        ~device();


        //////////////////////////////////////////////////
        // Local public functions
        cl_context getContext();
        

        //////////////////////////////////////////////////
        // Exposed functionality of this component
        //

        /// 
        clx_device_id test();
        void clCreateBuffer(cl_mem_flags, size_t);



    //[opencl_management_action_types
    HPX_DEFINE_COMPONENT_ACTION(device, test);
    HPX_DEFINE_COMPONENT_ACTION(device, clCreateBuffer);
    //]

    private:
        ///////////////////////////////////////////////
        // Private Member Functions
        //
        
        // Error Callback
        static void CL_CALLBACK error_callback(const char*, const void*,
                                               size_t, void*);

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
HPX_REGISTER_ACTION_DECLARATION(
       hpx::opencl::server::device::test_action,
    opencl_device_test_action);
HPX_REGISTER_ACTION_DECLARATION(
       hpx::opencl::server::device::clCreateBuffer_action,
    opencl_device_clCreateBuffer_action);
    
//]



#endif
