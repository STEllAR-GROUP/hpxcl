// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_BUFFER_HPP_
#define HPX_OPENCL_BUFFER_HPP_

// Default includes
#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

// Export definitions
#include "export_definitions.hpp"

// Forward Declarations
#include "fwd_declarations.hpp"

namespace hpx {
namespace opencl { 


    //////////////////////////////////////
    /// @brief Device memory.
    ///
    /// Every buffer belongs to one \ref device.
    ///
    class HPX_OPENCL_EXPORT buffer
      : public hpx::components::client_base<buffer, server::buffer>
    {
    
        typedef hpx::components::client_base<buffer, server::buffer> base_type;

        public:
            // Empty constructor, necessary for hpx purposes
            buffer(){}

            // Constructor
            buffer(hpx::shared_future<hpx::naming::id_type> const& gid)
              : base_type(gid)
            {}
            
            // ///////////////////////////////////////////////
            // Exposed Component functionality
            // 
 
            /**
             *  @brief Get the size of the buffer
             *  @return The size of the buffer
             */
            hpx::future<std::size_t>
            size() const;
            
    };

}}

#endif
