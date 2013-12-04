// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_DEVICE_HPP__
#define HPX_OPENCL_DEVICE_HPP__


#include <hpx/include/components.hpp>
#include <hpx/lcos/future.hpp>

#include "server/device.hpp"
#include "buffer.hpp"
#include "program.hpp"
#include <vector>


namespace hpx {
namespace opencl {

    

    class device
      : public hpx::components::client_base<
          device, hpx::components::stub_base<server::device>
        >
    
    {
    
        typedef hpx::components::client_base<
            device, hpx::components::stub_base<server::device>
            > base_type;

        public:
            device(){}

            device(hpx::future<hpx::naming::id_type> const& gid)
              : base_type(gid)
            {}
            
            //////////////////////////////////////////
            /// Exposed Component functionality
            /// 
            
            // Creates a user event
            hpx::lcos::future<hpx::opencl::event>
            create_user_event() const;
            
            
            // Creates an event that depends on a future
            template<class T>
            hpx::lcos::future<hpx::opencl::event>
            create_future_event(hpx::lcos::future<T> future); 

            // Creates an OpenCL buffer
            hpx::opencl::buffer
            create_buffer(cl_mem_flags flags, size_t size) const;
            hpx::opencl::buffer
            create_buffer(cl_mem_flags flags, size_t size, const void* data) const;

            // Creates an OpenCL program object
            hpx::opencl::program
            create_program_with_source(std::string source) const;
            
            
        private:
            //////////////////////////////////////////
            /// Helper Functions
            /// 
            
            // Needed for create_future_event, this is the future.then callback
            static void
            trigger_user_event_externally(hpx::lcos::future<hpx::opencl::event>);

    };


    template<class T>
    hpx::lcos::future<hpx::opencl::event>
    device::create_future_event(hpx::lcos::future<T> future)
    {
    
        // Create a user event
        hpx::lcos::future<hpx::opencl::event> event = create_user_event();
    
        // Schedule the user event trigger to be called after future
        future.then(
                hpx::util::bind(&(device::trigger_user_event_externally), event)
                        );
    
        // return the event
        return event;
    
    }

}}


#endif// HPX_OPENCL_DEVICE_HPP__
