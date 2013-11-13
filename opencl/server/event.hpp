// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_EVENT_HPP__
#define HPX_OPENCL_SERVER_EVENT_HPP__

#include <cstdint>

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>

#include <CL/cl.h>

// ! This header may NOT have dependencies to other components !
// A lot of components link to "../event.h", which links to this.
// Won't compile with recursive includes!

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{ namespace server{

    // Workaround to avoid including "device.hpp"
    class device;

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
        event(hpx::naming::id_type device_id, clx_event event_id);

        ~event();


        //////////////////////////////////////////////////
        // Local public functions
        //
        cl_event get_cl_event();
    
        //////////////////////////////////////////////////
        // Exposed functionality of this component
        //
        void await() const;

    //[opencl_management_action_types
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
        boost::shared_ptr<device> parent_device;
        cl_event event_id;
        hpx::naming::id_type parent_device_id;
    };
}}}

//[opencl_management_registration_declarations
HPX_REGISTER_ACTION_DECLARATION(
       hpx::opencl::server::event::await_action,
    opencl_event_await_action);
//]



#endif
