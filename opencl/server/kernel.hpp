// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_KERNEL_HPP
#define HPX_OPENCL_SERVER_KERNEL_HPP


#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "../cl_headers.hpp"

#include "../fwd_declarations.hpp"

// REGISTER_ACTION_DECLARATION templates
#include "util/server_definitions.hpp"

namespace hpx { namespace opencl{ namespace server{

    // /////////////////////////////////////////////////////
    //  This class represents an opencl kernel.

    class HPX_OPENCL_EXPORT kernel
      : public hpx::components::managed_component_base<kernel>
    {
    public:

        // Constructor
        kernel();
        // Destructor
        ~kernel();

        ///////////////////////////////////////////////////
        /// Local functions
        ///
        void init ( hpx::naming::id_type device_id, cl_program program,
                    std::string kernel_name );

        //////////////////////////////////////////////////
        /// Exposed functionality of this component
        ///

        // Sets an argument of the kernel
        void set_arg(cl_uint arg_index, hpx::naming::id_type buffer);

        // Runs the kernel
        void enqueue( hpx::naming::id_type && event_gid,
                      std::vector<std::size_t> size,
                      std::vector<hpx::naming::id_type> && dependencies );

    HPX_DEFINE_COMPONENT_ACTION(kernel, set_arg);
    HPX_DEFINE_COMPONENT_ACTION(kernel, enqueue);

        //////////////////////////////////////////////////
        // Private Member Functions
        //
    private:


        //////////////////////////////////////////////////
        //  Private Member Variables
        //
    private:
        std::shared_ptr<device> parent_device;
        cl_kernel kernel_id;
        hpx::naming::id_type parent_device_id;

    };

}}}

//[opencl_management_registration_declarations
HPX_OPENCL_REGISTER_ACTION_DECLARATION(kernel, set_arg);
HPX_OPENCL_REGISTER_ACTION_DECLARATION(kernel, enqueue);
//]

#endif
