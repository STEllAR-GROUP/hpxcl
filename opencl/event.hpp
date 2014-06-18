
// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_EVENT_HPP_
#define HPX_OPENCL_EVENT_HPP_

#include "server/event.hpp"

#include <hpx/include/components.hpp>
#include <hpx/lcos/future.hpp>


namespace hpx {
namespace opencl {

    /////////////////////
    /// @brief An OpenCL event.
    ///
    /// This is the main synchronization mechanism of this OpenCL framework.
    ///
    class event
      : public hpx::components::client_base<
          event, hpx::components::stub_base<server::event>
        >
    { 

        typedef hpx::components::client_base<
            event, hpx::components::stub_base<server::event>
            > base_type;

        public:
            // Empty constructor, necessary for hpx purposes
            event() {}

            // Constructor
            event(hpx::shared_future<hpx::naming::id_type> const& gid)
              : base_type(gid)
            {}

            // Converts hpx::opencl::event to cl_event
            static std::vector<cl_event>
            get_cl_events(std::vector<hpx::opencl::event>);
            static cl_event
            get_cl_event(hpx::opencl::event);

            // //////////////////////////////////////////////
            // Exposed functionality
            //
            
            /**
             *  @brief Blocks until the event has happened
             */
            void await() const;

            /**
             *  @brief Queries the state of the event.
             *
             *  @return True if the event already happened
             */
            hpx::lcos::future<bool> finished() const;

            /**
             *  @brief Converts the event to a hpx::lcos::future.
             *
             *  With this function and \ref device::create_future_event it is 
             *  possible to create inter-node-dependencies.
             *  
             *  @return A future that triggers when the event has happened.
             */
            hpx::lcos::future<void> get_future() const;

            /**
             *  @brief Triggers the event.
             *  
             *  This function can ONLY be called if the event is a user-event,
             *  created with \ref device::create_user_event.
             *
             *  Calling it on a non-user-event will result in undefined
             *  behaviour.
             */ 
            void trigger() const;

            /**
             *  @brief Retrieves the data associated with an event
             *
             *  With this method one can retrieve the data of an enqueue_read
             *  command.
             *
             *  @return The data.
             */
            hpx::lcos::future<boost::shared_ptr<std::vector<char>>>
            get_data() const;
    
    };

}}







#endif

