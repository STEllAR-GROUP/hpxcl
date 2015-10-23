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
#include "util/rect_props.hpp"

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
            {
                is_local =
                    (hpx::get_colocation_id_sync(get_id()) == hpx::find_here());
            }

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
             *  @brief Writes data to the buffer in a rectangular region
             *
             *  @param rect_properties  The parameters like size, offset, stride
             *  @param data             The data to be written.
             *
             *  @return        An future that can be used for synchronization or
             *                 dependency for other calls.
             */
            template<typename T, typename ...Deps>
            hpx::future<void>
            enqueue_write_rect(
                            rect_props rect_properties,
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
                          Deps &&... dependencies );

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

            /**
             *  @brief Reads data from the buffer
             *
             *  @param rect_properties      Parameters of the rectangle to read.
             *  @param data     The buffer the result will get written to.
             *                  The buffer will get returned and kept alive
             *                  through the future.
             *  @return         A future that can be used for synchronization or
             *                  dependency for other calls.
             *                  Contains the 'data' parameter with the result
             *                  written to.
             */
            template<typename T, typename ...Deps>
            hpx::future<hpx::serialization::serialize_buffer<T> >
            enqueue_read_rect( rect_props rect_properties,
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
            template<typename ...Deps>
            send_result enqueue_send( const hpx::opencl::buffer& dst,
                                      std::size_t         src_offset,
                                      std::size_t         dst_offset,
                                      std::size_t         size,
                                      Deps &&... dependencies );


            ////////////////////////////////////////////////////////////////////
            // Proxied functions
            //
        private:
            hpx::future<hpx::serialization::serialize_buffer<char> >
            enqueue_read_impl( std::size_t && offset,
                               std::size_t && size,
                               hpx::opencl::util::resolved_events && deps );

            send_result
            enqueue_send_impl( const hpx::opencl::buffer& dst,
                               std::size_t && src_offset,
                               std::size_t && dst_offset,
                               std::size_t && size,
                               hpx::opencl::util::resolved_events && deps );
        private:
            hpx::naming::id_type device_gid;
            bool is_local;

        private:
            // serialization support
            friend class hpx::serialization::access;

            template <typename Archive>
            void load(Archive & ar, unsigned)
            {
                ar >> hpx::serialization::base_object<base_type>(*this);
                ar >> device_gid;
                is_local =
                    (hpx::get_colocation_id_sync(get_id()) == hpx::find_here());
            }

            template <typename Archive>
            void save(Archive & ar, unsigned) const
            {
                ar << hpx::serialization::base_object<base_type>(*this);
                ar << device_gid;
            }

            HPX_SERIALIZATION_SPLIT_MEMBER()

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

    // send command to server class
    typedef hpx::opencl::server::buffer
        ::enqueue_read_to_userbuffer_local_action<T> func_local;
    typedef hpx::opencl::server::buffer
        ::enqueue_read_to_userbuffer_remote_action<T> func_remote;
    if(!is_local){
        // is remote call

        hpx::apply<func_remote>( std::move(get_id()),
                                 std::move(ev.get_event_id()),
                                 offset,
                                 data.size() * sizeof(T),
                                 reinterpret_cast<std::uintptr_t>
                                    ( data.data() ),
                                 std::move(deps.event_ids) );

    } else {
        // is local call, send direct reference to buffer

        hpx::apply<func_local>( std::move(get_id()),
                                std::move(ev.get_event_id()),
                                offset,
                                data,
                                std::move(deps.event_ids) );
    }

    // return future connected to event
    return ev.get_future();
}

template<typename T, typename ...Deps>
hpx::future<hpx::serialization::serialize_buffer<T> >
hpx::opencl::buffer::enqueue_read_rect(
                                   rect_props rect_properties,
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

    // send command to server class
    typedef hpx::opencl::server::buffer
        ::enqueue_read_to_userbuffer_rect_local_action<T> func_local;
    typedef hpx::opencl::server::buffer
        ::enqueue_read_to_userbuffer_rect_remote_action<T> func_remote;
    if(!is_local){
        // is remote call

        hpx::apply<func_remote>( std::move(get_id()),
                                 std::move(ev.get_event_id()),
                                 rect_properties,
                                 reinterpret_cast<std::uintptr_t>
                                    ( data.data() ),
                                 std::move(deps.event_ids) );

    } else {
        // is local call, send direct reference to buffer

        hpx::apply<func_local>( std::move(get_id()),
                                std::move(ev.get_event_id()),
                                rect_properties,
                                data,
                                std::move(deps.event_ids) );
    }

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
    hpx::apply<func>( this->get_id(),
                      ev.get_event_id(),
                      offset,
                      data,
                      std::move(deps.event_ids) );
                     

    // return future connected to event
    return ev.get_future();
}

template<typename T, typename ...Deps>
hpx::future<void>
hpx::opencl::buffer::enqueue_write_rect( rect_props rect_properties, 
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
    typedef hpx::opencl::server::buffer::enqueue_write_rect_action<T> func;
    hpx::apply<func>( this->get_id(),
                      ev.get_event_id(),
                      rect_properties,
                      data,
                      std::move(deps.event_ids) );

    // return future connected to event
    return ev.get_future();
}

template<typename ...Deps>
hpx::future<hpx::serialization::serialize_buffer<char> >
hpx::opencl::buffer::enqueue_read( std::size_t offset,
                                   std::size_t size,
                                   Deps &&... dependencies )
{
    // combine dependency futures in one std::vector
    using hpx::opencl::util::enqueue_overloads::resolver;
    auto deps = resolver(std::forward<Deps>(dependencies)...);
    HPX_ASSERT(deps.are_from_device(device_gid));

    return enqueue_read_impl( std::move(offset),
                              std::move(size),
                              std::move(deps) );
}

template<typename ...Deps>
hpx::opencl::buffer::send_result
hpx::opencl::buffer::enqueue_send( const hpx::opencl::buffer& dst,
                                   std::size_t src_offset,
                                   std::size_t dst_offset,
                                   std::size_t size,
                                   Deps &&... dependencies )
{
    // combine dependency futures in one std::vector
    using hpx::opencl::util::enqueue_overloads::resolver;
    auto deps = resolver(std::forward<Deps>(dependencies)...);

    return enqueue_send_impl( dst,
                              std::move(src_offset),
                              std::move(dst_offset),
                              std::move(size),
                              std::move(deps) );
}

#endif
