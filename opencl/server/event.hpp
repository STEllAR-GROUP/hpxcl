// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_EVENT_HPP__
#define HPX_OPENCL_SERVER_EVENT_HPP__

#include <cstdint>

#include <hpx/runtime/components/server/managed_component_base.hpp>

#include <CL/cl.h>

#include "device.hpp"

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{ namespace server{
    
    ////////////////////////////////////////////////////////
    /// This component wraps the cl_event type, to make it
    /// serializable and to add a global reference count.
    ///
    
    typedef intptr_t clx_event;

    class event
      : public hpx::components::managed_component_base<event>
    {
    public:
        // Constructor
        event();
        event(intptr_t device, clx_event event_id);

        ~event();


        //////////////////////////////////////////////////
        // Local public functions
        //
        cl_event get_cl_event();
    
        //////////////////////////////////////////////////
        // Exposed functionality of this component
        //
        hpx::naming::id_type get_future() const;
        void await() const;

    //[opencl_management_action_types
    HPX_DEFINE_COMPONENT_ACTION(event, get_future);
    HPX_DEFINE_COMPONENT_ACTION(event, await);
    //]

    private:
        ///////////////////////////////////////////////
        // Private Member Functions
        //
        

    private:
        ///////////////////////////////////////////////
        // Private Member Variables
        //
        device* parent_device;
        cl_event event_id;
    };
}}}

//[opencl_management_registration_declarations
HPX_REGISTER_ACTION_DECLARATION(
       hpx::opencl::server::event::get_future_action,
    opencl_event_get_future_action);
HPX_REGISTER_ACTION_DECLARATION(
       hpx::opencl::server::event::await_action,
    opencl_event_await_action);
//]



#endif
