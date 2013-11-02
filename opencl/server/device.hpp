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
#include "../event.hpp"

// ! This component header may NOT include other component headers !
// (To avoid recurcive includes)

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{ namespace server{
    
    ////////////////////////////////////////////////////////
    /// This class represents an OpenCL accelerator device.
    ///
    class device
      : public hpx::components::managed_component_base<device>
    {
    public:
        // Constructor
        device();
        device(clx_device_id device_id, bool enable_profiling=false);

        ~device();


        //////////////////////////////////////////////////
        /// Local public functions
        ///
        cl_context get_context();
        cl_command_queue get_read_command_queue();
        cl_command_queue get_write_command_queue();
        cl_command_queue get_work_command_queue();

        // Registers a read buffer
        void put_read_buffer(cl_event, boost::shared_ptr<std::vector<char>>);

        // Delete all ressources registered with specific cl_event
        void release_event_resources(cl_event);

        //////////////////////////////////////////////////
        /// Exposed functionality of this component
        ///
        
        // Creates an openCL Buffer
        hpx::naming::id_type
        clCreateBuffer(cl_mem_flags, size_t);
        
        // Returns the data associated with a certain cl_event
        boost::shared_ptr<std::vector<char>>
        get_event_data(hpx::opencl::event event_id);


    //[opencl_management_action_types
    HPX_DEFINE_COMPONENT_ACTION(device, clCreateBuffer);
    HPX_DEFINE_COMPONENT_ACTION(device, get_event_data);
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
        // Map for memory returned from readBuffer-Calls
        std::map<cl_event, boost::shared_ptr<std::vector<char>>> read_buffers;
        boost::mutex read_buffers_mutex;
    };
}}}

//[opencl_management_registration_declarations
HPX_REGISTER_ACTION_DECLARATION(
       hpx::opencl::server::device::clCreateBuffer_action,
    opencl_device_clCreateBuffer_action);
HPX_REGISTER_ACTION_DECLARATION(
       hpx::opencl::server::device::get_event_data_action,
    opencl_device_get_event_data_action);
    
//]



#endif
