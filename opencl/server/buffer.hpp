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


// REGISTER_ACTION_DECLARATION templates
#include "util/server_definitions.hpp"

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

        // Writes to the buffer
        template <typename T>
        void enqueue_write( hpx::naming::id_type && event_gid,
                            std::size_t offset,
                            hpx::serialization::serialize_buffer<T> data,
                            std::vector<hpx::naming::id_type> && dependencies );

        // Reads from the buffer
        void enqueue_read( hpx::naming::id_type && event_gid,
                           std::size_t offset,
                           std::size_t size,
                           std::vector<hpx::naming::id_type> && dependencies );


    HPX_DEFINE_COMPONENT_ACTION(buffer, size);
    HPX_DEFINE_COMPONENT_ACTION(buffer, enqueue_read);

    // Actions with template arguments (see enqueue_write<>() above) require
    // special type definitions. The simplest way to define such an action type
    // is by deriving from the HPX facility make_action:
    template <typename T>
    struct enqueue_write_action
      : hpx::actions::make_action<void (buffer::*)(
                        hpx::naming::id_type &&,
                        std::size_t,
                        hpx::serialization::serialize_buffer<T>,
                        std::vector<hpx::naming::id_type> &&),
            &buffer::template enqueue_write<T>, enqueue_write_action<T> >
    {};

    private:
        //////////////////////////////////////////////////
        //  Private Member Variables
        boost::shared_ptr<device> parent_device;
        cl_mem device_mem;
        hpx::naming::id_type parent_device_id;

    };

}}}

//[opencl_management_registration_declarations
HPX_OPENCL_REGISTER_ACTION_DECLARATION(buffer, size);
HPX_OPENCL_REGISTER_ACTION_DECLARATION(buffer, enqueue_read);
namespace hpx { namespace traits
{
    template <typename T>
    struct action_stacksize<hpx::opencl::server::buffer::enqueue_write_action<T> >
    {
        enum { value = hpx::threads::thread_stacksize_large };
    };
}}
//]


////////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATIONS
//

// HPXCL tools
#include "../tools.hpp"

// other hpxcl dependencies
#include "util/event_dependencies.hpp"
#include "device.hpp"


template <typename T>
void
hpx::opencl::server::buffer::enqueue_write(
                       hpx::naming::id_type && event_gid,
                       std::size_t offset,
                       hpx::serialization::serialize_buffer<T> data,
                       std::vector<hpx::naming::id_type> && dependencies ){
    
    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    cl_int err;
    cl_event return_event;

    // retrieve the dependency cl_events
    util::event_dependencies events( dependencies, parent_device.get() );

    // retrieve the command queue
    cl_command_queue command_queue = parent_device->get_write_command_queue();

    // run the OpenCL-call
    err = clEnqueueWriteBuffer( command_queue, device_mem, CL_FALSE, offset,
                                data.size()*sizeof(T), data.data(),
                                static_cast<cl_uint>(events.size()),
                                events.get_cl_events(), &return_event );
    cl_ensure(err, "clEnqueueWriteBuffer()");

    // register the data to prevent deallocation
    // TODO parent_device->put_event_data(return_event, data);

    // register the cl_event to the client event
    parent_device->register_event(event_gid, return_event);

}




#endif
