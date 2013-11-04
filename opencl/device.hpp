// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_DEVICE_HPP__
#define HPX_OPENCL_DEVICE_HPP__


#include <hpx/include/components.hpp>

#include "server/device.hpp"
#include "buffer.hpp"
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

            // Creates an OpenCL buffer
            hpx::opencl::buffer
            clCreateBuffer(cl_mem_flags flags, size_t size);
            hpx::opencl::buffer
            clCreateBuffer(cl_mem_flags flags, size_t size, const void* data);
            

            // Retrieves data associated with an event
            hpx::lcos::future<boost::shared_ptr<std::vector<char>>>
            get_event_data(hpx::opencl::event);


    };

}}



#endif// HPX_OPENCL_DEVICE_HPP__
