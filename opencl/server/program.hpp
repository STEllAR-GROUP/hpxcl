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

        //////////////////////////////////////////////////
        //  Private Member Variables
        //
    private:
        boost::shared_ptr<device> parent_device;
        cl_program program_cl;
        hpx::naming::id_type parent_device_id;

    };

}}}

//[opencl_management_registration_declarations
//]

#endif
