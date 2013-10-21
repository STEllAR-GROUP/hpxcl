// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_MEM_HPP__
#define HPX_OPENCL_SERVER_MEM_HPP__

#include <cstdint>

#include <hpx/include/iostreams.hpp>

#include <hpx/hpx_main.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>

#include <CL/cl.h>

#include "device.hpp"


// TODO synchronization

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{ namespace server{
    
    ///////////////////////////////////////////////////////////
    /// This class represents a generic opencl memory. (cl_mem)
    /// This is an abstract class and could hold an OpenCL buffer
    /// or an OpenCL image.
    ///
    class memory
      : public hpx::components::abstract_managed_component_base<memory>
    {
    public:
        // Constructor
        memory();
        memory(device *, size_t size);
        

        virtual ~memory() = 0;

        //////////////////////////////////////////////////
        // Exposed functionality of this component
        //
//f     virtual void releaseDeviceMem() = 0;
//f     virtual void regainDeviceMem() = 0;
        
        /////////////////////////////////////////////////
        // Functions with modified parameterlists for
        // easy access
        //         


        //////////////////////////////////////////
        // OLD CONCEPT - WILL NOT BE USED
        //
    //    virtual clx_event copyDeviceToHost(std::vector<clx_event>) = 0;
    //    virtual clx_event copyHostToDevice(std::vector<clx_event>) = 0;
    //    virtual void write(std::vector<char> mem, size_t offset) = 0;
    //    virtual std::vector<char> read(size_t size, size_t offset) = 0;
    //    clx_event         copyHostToDevice(clx_event);
    //    clx_event         copyDeviceToHost(clx_event);
    //    void              write(std::vector<char> mem) { write(mem, 0); }
    //    std::vector<char> read()                       { return read(size, 0); }

        


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
