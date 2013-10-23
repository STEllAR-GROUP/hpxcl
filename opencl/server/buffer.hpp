// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_BUFFER_HPP
#define HPX_OPENCL_SERVER_BUFFER_HPP


#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>

#include <CL/cl.h>


namespace hpx { namespace opencl{ namespace server{

    ////////////////////////////////////////////////////////
    /// This class represents an opencl buffer.
    
    class device;

    class buffer
      : public hpx::components::managed_component_base<buffer>
    {
    public:

        // Constructor
        buffer();
        buffer(intptr_t device, cl_mem_flags flags, size_t size,
               char* init_data = NULL);

        ~buffer();

        ///////////////////////////////////////////////////
        /// Exposed functionality of this component
        ///

    private:
        //////////////////////////////////////////////////
        /// Private Member Functions
        ///

    private:
        //////////////////////////////////////////////////
        //  Private Member Variables
        device* parent_device;
        size_t size;
        cl_mem device_mem;
    };



}}}






























#endif//HPX_OPENCL_SERVER_BUFFER_HPP

