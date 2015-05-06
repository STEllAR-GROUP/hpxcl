// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_BUFFER_HPP
#define HPX_OPENCL_SERVER_BUFFER_HPP


#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "../cl_headers.hpp"

#include "../fwd_declarations.hpp"


namespace hpx { namespace opencl{ namespace server{

    // /////////////////////////////////////////////////////
    //  This class represents an opencl buffer.

    class buffer
      : public hpx::components::managed_component_base<buffer>
    {
    public:

        // Constructor
        buffer();
        // Destructor
        ~buffer();

        ///////////////////////////////////////////////////
        /// Local functions
        /// 
        void init(hpx::naming::id_type device_id, cl_mem_flags flags,
                                                  std::size_t size);

        //////////////////////////////////////////////////
        /// Exposed functionality of this component
        ///
        // Returns the size of the buffer
        std::size_t size();

    HPX_DEFINE_COMPONENT_ACTION(buffer, size);
 
    private:
        //////////////////////////////////////////////////
        //  Private Member Variables
        boost::shared_ptr<device> parent_device;
        cl_mem device_mem;
        hpx::naming::id_type parent_device_id;

    };

}}}

//[opencl_management_registration_declarations
HPX_ACTION_USES_LARGE_STACK(hpx::opencl::server::buffer::size_action);
HPX_REGISTER_ACTION_DECLARATION(
        hpx::opencl::server::buffer::size_action,
        opencl_buffer_size_action);
//]

#endif
