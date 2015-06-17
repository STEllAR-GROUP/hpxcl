// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_BUFFER_HPP_
#define HPX_OPENCL_BUFFER_HPP_

// Default includes
#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

// Export definitions
#include "export_definitions.hpp"

// Forward Declarations
#include "fwd_declarations.hpp"

// Crazy function overloading
#include "util/enqueue_overloads.hpp"

#include "server/buffer.hpp"

namespace hpx {
namespace opencl { 


    //////////////////////////////////////
    /// @brief Device memory.
    ///
    /// Every buffer belongs to one \ref device.
    ///
    class HPX_OPENCL_EXPORT buffer
      : public hpx::components::client_base<buffer, server::buffer>
    {
    
        typedef hpx::components::client_base<buffer, server::buffer> base_type;

        public:
            // the result struct for enqueue_send
            struct send_result{
                public:
                    send_result( hpx::future<void>&& fut1,
                                 hpx::future<void>&& fut2 )
                        : src_future(std::move(fut1)),
                          dst_future(std::move(fut2)){};

                    hpx::future< void > src_future;
                    hpx::future< void > dst_future;
            };


        public:
            // Empty constructor, necessary for hpx purposes
            buffer(){}

            // Constructor
            buffer(hpx::shared_future<hpx::naming::id_type> const& gid,
                   hpx::naming::id_type device_gid_)
              : base_type(gid), device_gid(std::move(device_gid_))
            {}
            
            // initialization
            

            // ///////////////////////////////////////////////
            // Exposed Component functionality
            // 
 
            /**
             *  @brief Get the size of the buffer
             *
             *  @return The size of the buffer
             */
            hpx::future<std::size_t>
            size() const;

            /**
             *  @brief Writes data to the buffer
             *
             *  @param offset   The start position of the area to write to.
             *  @param data     The data to be written.
             *  @return         An future that can be used for synchronization or
             *                  dependency for other calls.
             */
            template<typename T, typename ...Deps>
            hpx::future<void>
            enqueue_write( std::size_t offset,
                           const hpx::serialization::serialize_buffer<T> data,
                           Deps &&... dependencies );

            /**
             *  @brief Reads data from the buffer
             *
             *  @param offset   The start position of the area to read.
             *  @param size     The size of the area to read.
             *  @return         A future that can be used for synchronization or
             *                  dependency for other calls.
             *                  Contains the result buffer of the call.
             */
            template<typename ...Deps>
            hpx::future<hpx::serialization::serialize_buffer<char> >
            enqueue_read( std::size_t offset,
                          std::size_t size,
                          Deps &&... dependencies )
            {
                return enqueue_read_alloc(std::move(offset), std::move(size),
                                          std::forward<Deps>(dependencies)...);
            }
            // This proxy function is necessary to prevent ambiguity with other
            // overloads
            HPX_OPENCL_GENERATE_ENQUEUE_OVERLOADS(
                hpx::future<hpx::serialization::serialize_buffer<char> >,
                                   enqueue_read_alloc, std::size_t /*offset*/,
                                                       std::size_t /*size*/);
         
            /**
             *  @brief Reads data from the buffer
             *
             *  @param offset   The start position of the area to read.
             *  @param data     The buffer the result will get written to.
             *                  The buffer also contains information about the
             *                  size of the data to read.
             *                  The buffer will get returned and kept alive
             *                  through the future.
             *  @return         A future that can be used for synchronization or
             *                  dependency for other calls.
             *                  Contains the 'data' parameter with the result
             *                  written to.
             */
            template<typename T, typename ...Deps>
            hpx::future<hpx::serialization::serialize_buffer<T> >
            enqueue_read( std::size_t offset,
                          hpx::serialization::serialize_buffer<T> data,
                          Deps &&... dependencies );

            /*
             *  @name Copies data to another buffer.
             *
             *  The buffers do NOT need to be from the same device,
             *  neither do they have to be on the same node.
             *
             *  @param dst          The source buffer.
             *  @param src_offset   The offset on the source buffer.
             *  @param dst_offset   The offset on the destination buffer.
             *  @param size         The size of the area to copy.
             *  @return             A future that can be used for synchronization
             *                      or dependency for other calls.
             *                  
             *  @see event
             */ 
            HPX_OPENCL_GENERATE_ENQUEUE_OVERLOADS(
                send_result, enqueue_send,
                                        const hpx::opencl::buffer& /*dst*/,
                                        std::size_t         /*src_offset*/,
                                        std::size_t         /*dst_offset*/,
                                        std::size_t         /*size*/ );
        private:
            hpx::naming::id_type device_gid;

    };

}}


////////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATIONS
//
template<typename T, typename ...Deps>
hpx::future<hpx::serialization::serialize_buffer<T> >
hpx::opencl::buffer::enqueue_read( std::size_t offset,
                                   hpx::serialization::serialize_buffer<T> data,
                                   Deps &&... dependencies )
{
    typedef hpx::serialization::serialize_buffer<T> buffer_type;

    // combine dependency futures in one std::vector
    using hpx::opencl::util::enqueue_overloads::resolver;
    auto deps = resolver(std::forward<Deps>(dependencies)...);
    HPX_ASSERT(deps.are_from_device(device_gid));
    
    // create local event
    using hpx::opencl::lcos::event;
    event<buffer_type> ev( device_gid, data );

    // asynchronously: 
    // check if the component is a on a different locality
    hpx::get_colocation_id(get_gid()).then(
        hpx::util::bind(
            []( hpx::future<hpx::naming::id_type>&& location,
                hpx::opencl::util::resolved_events& deps,
                std::size_t& offset,
                hpx::serialization::serialize_buffer<T>& data,
                hpx::naming::id_type& buffer_id,
                hpx::naming::id_type& event_id){

                // Check if this is a remote call
                bool is_remote_call =
                    (location.get() != hpx::find_here());

                // send command to server class
                typedef hpx::opencl::server::buffer
                    ::enqueue_read_to_userbuffer_local_action<T> func_local;
                typedef hpx::opencl::server::buffer
                    ::enqueue_read_to_userbuffer_remote_action<T> func_remote;
                if(is_remote_call){
                    // is remote call

                    hpx::apply<func_remote>( std::move(buffer_id),
                                             std::move(event_id),
                                             offset,
                                             data.size() * sizeof(T),
                                             reinterpret_cast<std::uintptr_t>
                                                ( data.data() ),
                                             deps.event_ids );
 
                } else {
                    // is local call, send direct reference to buffer

                    hpx::apply<func_local>( std::move(buffer_id),
                                            std::move(event_id),
                                            offset,
                                            data,
                                            deps.event_ids );
                }
              
            },
            hpx::util::placeholders::_1,
            std::move(deps),
            std::move(offset),
            std::move(data),
            this->get_gid(),
            ev.get_gid()
        )

    );

    // return future connected to event
    return ev.get_future();
}

template<typename T, typename ...Deps>
hpx::future<void>
hpx::opencl::buffer::enqueue_write( std::size_t offset,
               const hpx::serialization::serialize_buffer<T> data,
               Deps &&... dependencies )
{
    // combine dependency futures in one std::vector
    using hpx::opencl::util::enqueue_overloads::resolver;
    auto deps = resolver(std::forward<Deps>(dependencies)...);
    HPX_ASSERT(deps.are_from_device(device_gid));
    
    // create local event
    using hpx::opencl::lcos::event;
    event<void> ev( device_gid );

    // send command to server class
    typedef hpx::opencl::server::buffer::enqueue_write_action<T> func;
    hpx::apply<func>( this->get_gid(),
                      ev.get_gid(),
                      offset,
                      data,
                      deps.event_ids );
                     

    // return future connected to event
    return ev.get_future();
}




#endif
