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
                          Deps &&... dependencies )
            {
                typedef hpx::serialization::serialize_buffer<T> buffer_type;

                // combine dependency futures in one std::vector
                using hpx::opencl::util::enqueue_overloads::resolver;
                auto deps = resolver(std::forward<Deps>(dependencies)...);
                HPX_ASSERT(deps.are_from_device(device_gid));
                
                // check if the component is a on a different locality
                bool is_remote_call = false;
                if(hpx::get_colocation_id_sync(get_gid()) != find_here()){
                    is_remote_call = true;
                }

                // create local event
                using hpx::opencl::lcos::event;
                event<buffer_type> ev( device_gid, data );
            
                // send command to server class
                typedef hpx::opencl::server::buffer
                    ::enqueue_read_to_userbuffer_local_action<T> func_local;
                typedef hpx::opencl::server::buffer
                    ::enqueue_read_to_userbuffer_remote_action<T> func_remote;
                if(is_remote_call){
                    // is remote call

                    std::cout << "remote call!" << std::endl;
                    hpx::apply<func_remote>( this->get_gid(),
                                             ev.get_gid(),
                                             offset,
                                             data.size() * sizeof(T),
                                             reinterpret_cast<std::uintptr_t>
                                                ( data.data() ),
                                             deps.event_ids );
 
                } else {
                    // is local call, send direct reference to buffer

                    std::cout << "local call!" << std::endl;
                    hpx::apply<func_local>( this->get_gid(),
                                            ev.get_gid(),
                                            offset,
                                            data,
                                            deps.event_ids );
                }
           
                // return future connected to event
                return ev.get_future();
            }



        private:
            hpx::naming::id_type device_gid;

    };

}}

#endif
