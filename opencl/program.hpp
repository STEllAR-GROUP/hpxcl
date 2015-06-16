// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_PROGRAM_HPP_
#define HPX_OPENCL_PROGRAM_HPP_

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
    /// Every program belongs to one \ref device.
    ///
    class HPX_OPENCL_EXPORT program
      : public hpx::components::client_base<program, server::program>
    {
    
        typedef hpx::components::client_base<program, server::program> base_type;

        public:
            // Empty constructor, necessary for hpx purposes
            program(){}

            // Constructor
            program(hpx::shared_future<hpx::naming::id_type> const& gid,
                    hpx::naming::id_type device_gid_)
              : base_type(gid), device_gid(std::move(device_gid_))
            {}
            
            // initialization
            

            // ///////////////////////////////////////////////
            // Exposed Component functionality
            // 
 
            // Build the program, non-blocking
            /**
             *  @brief Builds the program, non-blocking.
             *
             *  @return A future that will trigger upon build completion.
             */
            hpx::lcos::future<void> build() const;
            /**
             *  @brief Builds the program, non-blocking.
             *
             *  @param build_options    A string with specific build options.<BR>
             *                          Look at the official 
             *                          <A HREF="http://www.khronos.org/registry
             * /cl/sdk/1.2/docs/man/xhtml/clBuildProgram.html">
             *                          OpenCL Reference</A> for further
             *                          information.
             *  @return A future that will trigger upon build completion.
             */
            hpx::lcos::future<void> build(std::string build_options) const;

        private:
            hpx::naming::id_type device_gid;

    };

}}

#endif
