// Copyright (c)    2015 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_UTIL_EVENT_MAP_HPP_
#define HPX_OPENCL_SERVER_UTIL_EVENT_MAP_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "../../cl_headers.hpp"

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{ namespace server{ namespace util{
    

    ////////////////////////////////////////////////////////
    // This class is used for the mapping between gid's and cl_events.
    //
    class event_map
    {
        typedef hpx::lcos::local::spinlock lock_type;
        typedef hpx::lcos::local::condition_variable condition_type;

    public:
        // Constructor
        event_map();
        ~event_map();

        //////////////////////////////////////////////////
        /// Local public functions
        ///

        // Adds a new gid-cl_event pair
        // !! add does not have any sequential consistency guarantee to get().
        // It might happen that get() gets called before the referencing GID
        // is registered with add().
        // (=> Fancy synchronization needed inside of event_map)
        void add(const hpx::naming::id_type&, cl_event);

        // Retrieves the cl_event associated with the gid.
        // !! BLOCKS if gid is not present until gid gets added
        // with 'add()'.
        cl_event get(const hpx::naming::id_type&);

        // Registers a function that will get called upon gid removal
        // (Used to delete associated cl_event)
        void register_deletion_callback(std::function<void(cl_event)> &&);

        // Removes a GID.
        // !! This function is the only one of this class with a consistency
        // guarantee. remove() will ALWAYS be called AFTER all other calls
        // involving the given GID are finished. (i.e. add() and get())
        void remove(const hpx::naming::gid_type&);

    private:
        ///////////////////////////////////////////////
        // Private Member Variables
        //

        // The actual internal datastructure
        typedef std::map<hpx::naming::gid_type, cl_event>
            map_type;
        map_type events;

        // Threads that called get() and are waiting for a corresponding add()
        typedef std::map<hpx::naming::gid_type, std::shared_ptr<condition_type> >
            waitmap_type;
        waitmap_type waits;

        // Lock for synchronization
        lock_type lock;

        // Callback function for cl_event cleanup
        std::function<void(cl_event)> deletion_callback;

    };
}}}}

#endif
