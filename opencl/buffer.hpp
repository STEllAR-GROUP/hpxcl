// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_BUFFER_HPP__
#define HPX_OPENCL_BUFFER_HPP__


#include <hpx/include/components.hpp>
#include <hpx/lcos/future.hpp>

#include "server/buffer.hpp"
#include "clx_event.hpp"
#include <vector>

namespace hpx {
namespace opencl { 
    

    class buffer
      : public hpx::components::client_base<
          buffer, hpx::components::stub_base<server::buffer>
        >
    {
    
        typedef hpx::components::client_base<
            buffer, hpx::components::stub_base<server::buffer>
            > base_type;

        public:
            // Empty constructor, necessary for hpx purposes
            buffer(){}

            // Constructor
            buffer(hpx::future<hpx::naming::id_type> const& gid)
              : base_type(gid)
            {}
            
            /////////////////////////////////////////////////
            /// Exposed Component functionality
            /// 
            hpx::lcos::future<clx_event>
            clEnqueueReadBuffer(size_t offset, size_t size, bool ptr,
                                          std::vector<clx_event> event);

    };

}}



#endif// HPX_OPENCL_BUFFER_HPP__
