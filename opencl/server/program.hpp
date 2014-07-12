// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_PROGRAM_HPP_
#define HPX_OPENCL_SERVER_PROGRAM_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>
#include <hpx/include/components.hpp>
#include <hpx/util/serialize_buffer.hpp>

#include <CL/cl.h>

#include "../fwd_declarations.hpp"

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{ namespace server{

    // /////////////////////////////////////////////////////
    //  This component represents an OpenCL program object.
    // 
    
    class program
      : public hpx::components::managed_component_base<program>
    {
    public:
        // Constructor
        program();
        program(hpx::naming::id_type device_id, std::string code);
        program(hpx::naming::id_type device_id,
                                   hpx::util::serialize_buffer<char> binary);

        ~program();

        //////////////////////////////////////////////////
        /// Local functions
        /// 
        cl_program get_cl_program();
        hpx::naming::id_type get_device_id();

        //////////////////////////////////////////////////
        /// Exposed functionality of this component
        ///
        
        // Compiles and links the program
        //      (mutually exclusive to compile() and link())
        void build(std::string options);

        // Compiles and links seperately
        //      (mutually exclusive to build())
//      void compile(std::string options,
//                   std::vector<hpx::opencl::header> headers);
//      void link(std::string options,
//                std::vector<hpx::opencl::program> dependencies);

        // Returns the binary representation of the program
        std::vector<char> get_binary();

    //[opencl_management_action_types
    HPX_DEFINE_COMPONENT_ACTION(program, build);
    HPX_DEFINE_COMPONENT_ACTION(program, get_binary);
    //]

    private:
        ///////////////////////////////////////////////
        /// Private Member Functions
        ///

        // returns the build log
        std::string acquire_build_log(); 
    
        // checks for build errors
        void throw_on_build_errors(cl_device_id device_id,
                                   const char* function_name);

    private:
        ///////////////////////////////////////////////
        // Private Member Variables
        //
        boost::shared_ptr<device> parent_device;
        hpx::naming::id_type parent_device_id;

        // the cl_program object
        cl_program program_id;

    };
}}}

//[opencl_management_registration_declarations
HPX_REGISTER_ACTION_DECLARATION(
       hpx::opencl::server::program::build_action,
    opencl_program_build_action);
HPX_REGISTER_ACTION_DECLARATION(
       hpx::opencl::server::program::get_binary_action,
    opencl_program_get_binary_action);
//]



#endif
