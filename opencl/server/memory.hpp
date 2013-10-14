// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_MEM_HPP__
#define HPX_OPENCL_SERVER_MEM_HPP__

#include <cstdint>

#include <hpx/include/iostreams.hpp>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>

#include <CL/cl.h>

#include "device.hpp"

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{ namespace server{
    
    ////////////////////////////////////////////////////////
    /// This class represents an OpenCL accelerator device.
    ///
    class memory
      : public hpx::components::locking_hook<
          hpx::components::managed_component_base<memory>
        >
    {
    public:
        // Constructor
        memory();
        memory(device *, size_t size);
        

        virtual ~memory() = 0;

        //////////////////////////////////////////////////
        // Exposed functionality of this component
        //



    private:
        ///////////////////////////////////////////////
        // Private Member Functions
        //
        

    private:
        ///////////////////////////////////////////////
        // Private Member Variables
        //
        device *parent_device;
        cl_mem device_mem;
        std::vector<char> host_mem;
        size_t size;

    };
}}}

#endif
