// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_PROGRAM_HPP__
#define HPX_OPENCL_SERVER_PROGRAM_HPP__

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>

#include <CL/cl.h>

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{ namespace server{

    class device;

    ////////////////////////////////////////////////////////
    /// This component represents an OpenCL program object.
    ///
    
    class program
      : public hpx::components::managed_component_base<program>
    {
    public:
        // Constructor
        program();
        program(hpx::naming::id_type device_id, std::string code);

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


    //[opencl_management_action_types
    HPX_DEFINE_COMPONENT_ACTION(program, build);
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

        // the program code
        std::string code;

        // the cl_program object
        cl_program program_id;

    };
}}}

//[opencl_management_registration_declarations
HPX_REGISTER_ACTION_DECLARATION(
       hpx::opencl::server::program::build_action,
    opencl_program_build_action);
//]



#endif
