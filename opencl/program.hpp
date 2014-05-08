// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_PROGRAM_HPP__
#define HPX_OPENCL_PROGRAM_HPP__

#include "server/program.hpp"

#include <hpx/include/components.hpp>

#include "fwd_declarations.hpp"

namespace hpx {
namespace opencl { 

    //////////////////////////////////////
    /// @brief A collection of kernels.
    /// 
    /// One program represents one or multiple kernels.
    /// It can be created from one (TODO multiple) source files.
    ///
    class program
      : public hpx::components::client_base<
          program, hpx::components::stub_base<server::program>
        >
    {
    
        typedef hpx::components::client_base<
            program, hpx::components::stub_base<server::program>
            > base_type;

        public:
            // Empty constructor, necessary for hpx purposes
            program(){}

            // Constructor
            program(hpx::shared_future<hpx::naming::id_type> const& gid)
              : base_type(gid)
            {}
            
            /////////////////////////////////////////////////
            /// Exposed Component functionality
            /// 
            
            // Build the Program, blocking
            /**
             *  @brief Builds the program, blocking.
             */
            void build() const;
            /**
             *  @brief Builds the program, blocking.
             *
             *  @param build_options    A string with specific build options.<BR>
             *                          Look at the official 
             *                          <A HREF="http://www.khronos.org/registry
             * /cl/sdk/1.2/docs/man/xhtml/clBuildProgram.html">
             *                          OpenCL Reference</A> for further
             *                          information.
             */
            void build(std::string build_options) const;

            // Build the program, non-blocking
            /**
             *  @brief Builds the program, non-blocking.
             *
             *  @return A future that will trigger upon build completion.
             */
            hpx::lcos::future<void> build_async() const;
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
            hpx::lcos::future<void> build_async(std::string build_options) const;

            // Get the binary of a built program
            /**
             *  @brief
             *
             *
             *
             */
            hpx::lcos::future<std::vector<char>> get_binary() const;

            /**
             *  @brief Creates a kernel.
             *
             *  The kernel with the name kernel_name has to be defined and
             *  implemented in the program source code.
             *
             *  @param kernel_name  The name of the kernel to be created
             *  @return             A kernel object.
             */
            hpx::opencl::kernel
            create_kernel(std::string kernel_name) const;

    };

}}



#endif
