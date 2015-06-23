// Copyright (c)    2015 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_UTIL_DATA_MAP_HPP_
#define HPX_OPENCL_SERVER_UTIL_DATA_MAP_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "../../cl_headers.hpp"

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{ namespace server{ namespace util{
    

    ////////////////////////////////////////////////////////
    // This class is used to hide the template parameter from serialize_buffer.
    //
    class data_map_entry
    {
    private:
        template <typename T, typename Alloc>
        static void
        send_to_client_impl( hpx::serialization::serialize_buffer<T, Alloc> data,
                             const hpx::naming::id_type& event_id )
        {
            hpx::set_lco_value(event_id, data, false); 
        }

    public:
        template <typename T, typename Alloc>
        void set_data(hpx::serialization::serialize_buffer<T, Alloc> data)
        {

            // The data itself does not need to explicitely get kept alive,
            // it gets kept alive inside of the bind.
            
            send_callback = hpx::util::bind(&send_to_client_impl<T, Alloc>, data,
                                             hpx::util::placeholders::_1);
        }

        // Sends the data to the client event (to trigger client future)
        void send_data_to_client(const hpx::naming::id_type& client_event);
        
    private:
        hpx::util::function_nonser<void(const hpx::naming::id_type&)>
            send_callback;
    };


    ////////////////////////////////////////////////////////
    // This class is used for keeping data associated with cl_events alive.
    //
    class data_map
    {
        typedef hpx::lcos::local::spinlock lock_type;
    public:
        // Constructor
        data_map();
        ~data_map();

        //////////////////////////////////////////////////
        /// Local public functions
        ///

        // Registers a data chunk to a cl_event
        template <typename T, typename Alloc>
        void add( cl_event event,
                  hpx::serialization::serialize_buffer<T, Alloc> data )
        {
            // Strip the template from the buffer
            data_map_entry entry;
            entry.set_data(data);

            {
                // Lock the map
                lock_type::scoped_lock l(lock);

                // Insert the data into the map
                map.insert(std::move(
                        map_type::value_type(event, std::move(entry))
                    ));
            }
        }
     
        // Returns the data entry associated with the event.
        // Undefined behaviour if no data is available.
        data_map_entry get(cl_event event);

        // Returns bool if data is registered, and false if not
        bool has_data(cl_event event);

        // Deletes the data
        void remove(cl_event event);

    private:
        ///////////////////////////////////////////////
        // Private Member Variables
        //

        // The actual internal datastructure
        typedef std::map<cl_event, data_map_entry> map_type;
        map_type map;

        // Lock for synchronization
        lock_type lock;

    };
}}}}

#endif
