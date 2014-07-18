// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_KERNEL_HPP_
#define HPX_OPENCL_SERVER_KERNEL_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>
#include <hpx/include/components.hpp>

#include <CL/cl.h>

#include "../fwd_declarations.hpp"
#include "../event.hpp"

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{ namespace server{

    // /////////////////////////////////////////////////////
    //  This component represents an OpenCL program object.
    // 
    
    class kernel
      : public hpx::components::managed_component_base<kernel>
    {
    public:
        // Constructor
        kernel();
        kernel(hpx::naming::id_type program_id, std::string kernel_name);
        ~kernel();


        //////////////////////////////////////////////////
        /// Exposed functionality of this component
        ///

        // Sets an argument of the kernel
        void set_arg(cl_uint arg_index, hpx::opencl::buffer arg);

        // Runs the kernel
        hpx::opencl::event
        enqueue(cl_uint work_dim, std::vector<std::vector<size_t>> args,
                                  std::vector<hpx::opencl::event> events);

    //[opencl_management_action_types
    HPX_DEFINE_COMPONENT_ACTION(kernel, set_arg);
    HPX_DEFINE_COMPONENT_ACTION(kernel, enqueue);
    //]

    private:
        ///////////////////////////////////////////////
        // Private Member Functions
        //

    private:
        ///////////////////////////////////////////////
        // Private Member Variables
        //
        boost::shared_ptr<program> parent_program;
        hpx::naming::id_type       parent_program_id;
        boost::shared_ptr<device>  parent_device;
        hpx::naming::id_type       parent_device_id;

        // the cl_kernel object
        cl_kernel kernel_id;

    };
}}}

//[opencl_management_registration_declarations
HPX_REGISTER_ACTION_DECLARATION(
        hpx::opencl::server::kernel::set_arg_action,
        opencl_kernel_set_arg_action);
HPX_REGISTER_ACTION_DECLARATION(
        hpx::opencl::server::kernel::enqueue_action,
        opencl_kernel_enqueue_action);
//]



#endif
