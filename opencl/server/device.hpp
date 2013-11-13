// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_DEVICE_HPP__
#define HPX_OPENCL_SERVER_DEVICE_HPP__

#include <cstdint>

#include <hpx/include/iostreams.hpp>
#include <hpx/util/serialize_buffer.hpp>

#include <hpx/runtime/components/server/managed_component_base.hpp>


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
        cl_device_id get_device_id();
        cl_command_queue get_read_command_queue();
        cl_command_queue get_write_command_queue();
        cl_command_queue get_work_command_queue();

        // Registers a read buffer
        void put_event_data(cl_event, boost::shared_ptr<std::vector<char>>);

        // Delete all ressources registered with specific cl_event
        void release_event_resources(cl_event);

        // Returns the data associated with a certain cl_event
        boost::shared_ptr<std::vector<char>>
        get_event_data(cl_event event);
        
        //////////////////////////////////////////////////
        /// Exposed functionality of this component
        ///
        

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
        // Map for data returned from opencl calls
        std::map<cl_event, boost::shared_ptr<std::vector<char>>> event_data;
        boost::mutex event_data_mutex;
    };
}}}

//[opencl_management_registration_declarations
//]



#endif
