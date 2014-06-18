// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_DEVICE_HPP_
#define HPX_OPENCL_DEVICE_HPP_

#include "server/device.hpp"

#include <hpx/include/components.hpp>
#include <hpx/lcos/future.hpp>

#include <vector>

#include "fwd_declarations.hpp"

namespace hpx {
namespace opencl {

    /////////////////////////////////////////
    /// @brief An accelerator device.
    ///
    class device
      : public hpx::components::client_base<
          device, hpx::components::stub_base<server::device>
        >
    
    {
    
        typedef hpx::components::client_base<
            device, hpx::components::stub_base<server::device>
            > base_type;

        public:
            device(){}

            device(hpx::shared_future<hpx::naming::id_type> const& gid)
              : base_type(gid)
            {}
            
            // ///////////////////////////////////////
            // Exposed Component functionality
            // 
            
            /**
             *  @brief Creates a user event
             *  
             *  User events can be triggered with \ref event::trigger().
             *
             *  @return The user event.
             */
            hpx::lcos::future<hpx::opencl::event>
            create_user_event() const;
            
            /**
             *  @brief Queries device infos.
             *  
             *  @param info_type    The type of information.<BR>
             *                      A complete list can be found on the official
             *                      <A HREF="http://www.khronos.org/registry/cl/
             * sdk/1.2/docs/man/xhtml/clGetDeviceInfo.html">
             *                      OpenCL Reference</A>.
             *  @return The info data as char array.<BR>
             *          This will typically be cast to some other type via
             *          (for example):
             *          \code{.cpp}
             *          cl_uint *return_uint = (cl_uint*)&return_charvector[0];
             *          \endcode
             *          or converted to a string via \ref device_info_to_string.
             */
            hpx::lcos::future<std::vector<char>>
            get_device_info(cl_device_info info_type) const;
            
            /** 
             *  @brief Converts device info data to a string
             *
             *  This method is for convenience.<BR>
             *  It should only be used on String return types of
             *  \ref get_device_info.
             *
             *  @param info     Output data of \ref get_device_info. <BR>
             *                  Only use this function if the data type is
             *                  a string.
             *  @return         The data, converted to a string.
             */
            static std::string
            device_info_to_string(hpx::lcos::future<std::vector<char>> info);
            
            /**
             *  @brief Creates an event that triggers on the completion of
             *         a future.
             *
             *  This function is an essential tool for the interoperability of
             *  events and futures.
             *
             *  @param future   An hpx::lcos::future.
             *  @return         An event that will trigger as soon as the
             *                  future is completed.
             */
            template<class T>
            hpx::lcos::shared_future<hpx::opencl::event>
            create_future_event(hpx::lcos::shared_future<T> & future); 
            template<class T>
            hpx::lcos::shared_future<hpx::opencl::event>
            create_future_event(hpx::lcos::future<T> && future); 

            /**
             *  @brief Creates an OpenCL buffer.
             *
             *  @param flags    Sets properties of the buffer.<BR>
             *                  Possible values are
             *                      - CL_MEM_READ_WRITE
             *                      - CL_MEM_WRITE_ONLY
             *                      - CL_MEM_READ_ONLY
             *                      - CL_MEM_HOST_WRITE_ONLY
             *                      - CL_MEM_HOST_READ_ONLY
             *                      - CL_MEM_HOST_NO_ACCESS
             *                      .
             *                  and combinations of them.<BR>
             *                  For further information, read the official
             *                  <A HREF="http://www.khronos.org/registry/cl/sdk/
             * 1.2/docs/man/xhtml/clCreateBuffer.html">
             *                  OpenCL Reference</A>.
             *  @param size     The size of the buffer, in bytes.
             *  @return         A new \ref buffer object.
             *  @see            buffer
             */
            // Creates an OpenCL buffer
            hpx::opencl::buffer
            create_buffer(cl_mem_flags flags, size_t size) const;

            /**
             *  @brief Creates an OpenCL buffer and initializes it with given
             *         data.
             *
             *  The data pointer must NOT get modified until the internal
             *  future of the buffer triggered.
             *
             *  One can wait for the internal future with e.g. buffer::get_gid().
             *
             *  @param flags    Sets properties of the buffer.<BR>
             *                  Possible values are
             *                      - CL_MEM_READ_WRITE
             *                      - CL_MEM_WRITE_ONLY
             *                      - CL_MEM_READ_ONLY
             *                      - CL_MEM_HOST_WRITE_ONLY
             *                      - CL_MEM_HOST_READ_ONLY
             *                      - CL_MEM_HOST_NO_ACCESS
             *                      .
             *                  and combinations of them.<BR>
             *                  For further information, read the official
             *                  <A HREF="http://www.khronos.org/registry/cl/sdk/
             * 1.2/docs/man/xhtml/clCreateBuffer.html">
             *                  OpenCL Reference</A>.
             *  @param size     The size of the buffer, in bytes.
             *  @param data     The initialization data.
             *  @return         A new \ref buffer object that contains the given
             *                  data.
             *  @see            buffer
             */
            hpx::opencl::buffer
            create_buffer(cl_mem_flags flags, size_t size, const void* data) const;

            /**
             *  @brief Creates an OpenCL program object
             *  
             *  After creating a program object, one usually compiles the
             *  program an creates kernels from it.
             *
             *  One program can contain code for multiple kernels.
             *
             *  @param source   The source code string for the program.
             *  @return         A program object associated with the calling
             *                  device.
             */             
            hpx::opencl::program
            create_program_with_source(std::string source) const;
            
            /**
             *  @brief Creates an OpenCL program object from a prebuilt binary
             *
             *  One can create a prebuilt binary from a compiled
             *  \ref hpx::opencl::program with \ref program::get_binary()
             *
             *  @param binary   The binary execution code for the program.
             *  @return         A program object associated with the calling
             *                  device.
             */
            hpx::opencl::program
            create_program_with_binary(size_t binary_size, const char* binary) const;

        private:
            // ///////////////////////////////////////
            //  Helper Functions
            //  
            
            // Needed for create_future_event, this is the future.then callback
            static void
            trigger_user_event_externally(hpx::lcos::shared_future<hpx::opencl::event>);

    };


    template<class T>
    hpx::lcos::shared_future<hpx::opencl::event>
    device::create_future_event(hpx::lcos::shared_future<T> & future)
    {
    
        // Create a user event
        hpx::lcos::shared_future<hpx::opencl::event> event = create_user_event();
    
        // Schedule the user event trigger to be called after future
        future.then(
                       hpx::util::bind(&(device::trigger_user_event_externally),
                                       event)
                   );

        // Return the event
        return event;

    }

    template<class T>
    hpx::lcos::shared_future<hpx::opencl::event>
    device::create_future_event(hpx::lcos::future<T> && future)
    {
    
        // Create a user event
        hpx::lcos::shared_future<hpx::opencl::event> event = create_user_event();
    
        // Schedule the user event trigger to be called after future
        future.then(
                       hpx::util::bind(&(device::trigger_user_event_externally),
                                       event)
                   );

        // Return the event
        return event;

    }

}}


#endif// HPX_OPENCL_DEVICE_HPP_
