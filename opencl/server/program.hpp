// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_PROGRAM_HPP
#define HPX_OPENCL_SERVER_PROGRAM_HPP


#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "../cl_headers.hpp"

#include "../fwd_declarations.hpp"

// REGISTER_ACTION_DECLARATION templates
#include "util/server_definitions.hpp"

namespace hpx { namespace opencl{ namespace server{

    // /////////////////////////////////////////////////////
    //  This class represents an opencl program.

    class program
      : public hpx::components::managed_component_base<program>
    {
    public:

        // Constructor
        program();
        // Destructor
        ~program();

        ///////////////////////////////////////////////////
        /// Local functions
        /// 
        void init_with_source( hpx::naming::id_type device_id,
                               hpx::serialization::serialize_buffer<char> src);

        //////////////////////////////////////////////////
        /// Exposed functionality of this component
        ///

        // builds the program.
        // mutually exclusive to compile() and link().
        void build(std::string options);

        // Returns the binary representation of the program
        hpx::serialization::serialize_buffer<char> get_binary();

        // creates a kernel from the buffer
        hpx::naming::id_type create_kernel(std::string kernel_name);

    HPX_DEFINE_COMPONENT_ACTION(program, build);
    HPX_DEFINE_COMPONENT_ACTION(program, get_binary);
    HPX_DEFINE_COMPONENT_ACTION(program, create_kernel);

        //////////////////////////////////////////////////
        // Private Member Functions
        //
    private:

        // returns the build log
        std::string acquire_build_log();

        // checks for build errors
        void throw_on_build_errors(const char* function_name);


        //////////////////////////////////////////////////
        //  Private Member Variables
        //
    private:
        boost::shared_ptr<device> parent_device;
        cl_program program_id;
        hpx::naming::id_type parent_device_id;

    };

}}}

//[opencl_management_registration_declarations
HPX_OPENCL_REGISTER_ACTION_DECLARATION(program, build);
HPX_OPENCL_REGISTER_ACTION_DECLARATION(program, get_binary);
HPX_OPENCL_REGISTER_ACTION_DECLARATION(program, create_kernel);
//]

#endif
