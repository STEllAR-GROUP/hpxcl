// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_KERNEL_HPP_
#define HPX_OPENCL_KERNEL_HPP_

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
    /// Every kernel belongs to one \ref device.
    ///
    class HPX_OPENCL_EXPORT kernel
      : public hpx::components::client_base<kernel, server::kernel>
    {
    
        typedef hpx::components::client_base<kernel, server::kernel> base_type;

        public:
            // Empty constructor, necessary for hpx purposes
            kernel(){}

            // Constructor
            kernel(hpx::shared_future<hpx::naming::id_type> const& gid,
                    hpx::naming::id_type device_gid_)
              : base_type(gid), device_gid(std::move(device_gid_))
            {}
            
            // initialization
            

            // ///////////////////////////////////////////////
            // Exposed Component functionality
            // 
 
        private:
            hpx::naming::id_type device_gid;

    };

}}

#endif
