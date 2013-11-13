
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
            get_cl_events(hpx::opencl::event);

            void await();
            hpx::lcos::future<void> get_future();

    };

}}







#endif

