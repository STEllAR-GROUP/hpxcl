
// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_EVENT_HPP__
#define HPX_OPENCL_EVENT_HPP__

#include <hpx/include/components.hpp>
#include <hpx/lcos/future.hpp>

#include "server/event.hpp"

namespace hpx {
namespace opencl {


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
            event(hpx::future<hpx::naming::id_type> const& gid)
              : base_type(gid)
            {}

            // Converts hpx::opencl::event to cl_event
            static std::vector<cl_event>
            get_cl_events(std::vector<hpx::opencl::event>);
            static cl_event
            get_cl_event(hpx::opencl::event);

            // Blocks until the cl_event has happened
            void await() const;

            // Returns true if the event already happened
            hpx::lcos::future<bool> finished() const;

            // Returns a future variable that triggers when the cl_event has 
            // happened
            hpx::lcos::future<void> get_future() const;

            // Triggers the event. This call is only valid if the event is a 
            // user-created event. Calling this function if the event is not
            // user-created will result in undefined behaviour.
            void trigger() const;

            // Retrieves the data associated with the event
            hpx::lcos::future<boost::shared_ptr<std::vector<char>>>
            get_data() const;
    
    };

}}







#endif

