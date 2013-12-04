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
#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <hpx/runtime/components/server/managed_component_base.hpp>

#include <queue>
#include <map>

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

        // blocks until event triggers
        void wait_for_event(cl_event event);
        
        // Schedules mem object for deletion
        //
        // this is a workaround for the clSetEventStatus <-> clReleaseMemObj
        // problem; further information:
        // http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clSetUserEventStatus.html
        // 
        void schedule_cl_mem_deletion(cl_mem mem);

        // triggers an event previously generated with create_user_event()
        void trigger_user_event(cl_event event);
        
        //////////////////////////////////////////////////
        /// Exposed functionality of this component
        ///
        
        // creates an opencl event that can be triggered by the user
        hpx::opencl::event create_user_event();


    HPX_DEFINE_COMPONENT_ACTION(device, create_user_event);

    private:
        ///////////////////////////////////////////////
        // Private Member Functions
        //
        
        // Error Callback
        static void CL_CALLBACK error_callback(const char*, const void*,
                                               size_t, void*);
        // Try to delete buffers
        void try_delete_cl_mem();
        // same as above, but doesn't lock user_events_mutex internally.
        // calling function needs to assure that user_events_mutex is already
        // locked.
        void try_delete_cl_mem_nolock();
        
        // same as trigger_user_event, but doesn't lock user_events_mutex.
        // calling function needs to lock user_events_mutex manually.
        void trigger_user_event_nolock(cl_event);

        // cleans up all the possible leftover user events an cl_mems
        void cleanup_user_events();


    private:
        ///////////////////////////////////////////////
        // Private Member Variables
        //
        cl_device_id        device_id;
        cl_platform_id      platform_id;
        cl_context          context;
        cl_command_queue    command_queue;

        // lock typedefs
        typedef hpx::lcos::local::mutex mutex_type;
        typedef hpx::lcos::local::spinlock spinlock_type;

        // Map for data returned from opencl calls
        // (e.g. from buffer::enqueue_read)
        std::map<cl_event, boost::shared_ptr<std::vector<char>>> event_data;
        spinlock_type event_data_mutex;
        
        // List for all the user generated events (e.g. from futures)
        // Store hpx::opencl::event client with them to keep reference counter up
        std::map<cl_event, hpx::opencl::event> user_events;
        spinlock_type user_events_mutex;

        // List of pending cl_mem deletions
        // this is a workaround for the clSetEventStatus problem
        std::queue<cl_mem> pending_cl_mem_deletions; 
        spinlock_type pending_cl_mem_deletions_mutex;

        // List of waiting events with respective mutexes
        std::map<cl_event, boost::shared_ptr<hpx::lcos::local::event>>
                                                            cl_event_waitlist;
        spinlock_type cl_event_waitlist_mutex;

    };
}}}

//[opencl_management_registration_declarations
HPX_REGISTER_ACTION_DECLARATION(
        hpx::opencl::server::device::create_user_event_action,
        opencl_device_create_user_event_action);
//]



#endif
