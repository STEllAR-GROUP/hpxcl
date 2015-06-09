// Copyright (c)    2015 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The Header of this class
#include "data_map.hpp"


using hpx::opencl::server::util::data_map;
using hpx::opencl::server::util::data_map_entry;

data_map::data_map(){

}

data_map::~data_map(){
    // Correct use removes all entries from this map before deletion
    HPX_ASSERT(map.empty());
}



void 
data_map::remove(cl_event event)
{
    // Lock
    lock_type::scoped_lock l(lock);

    // Remove element
    map.erase(event);
}



void 
data_map::send_data_to_client(const hpx::naming::id_type& client_event,
                              cl_event event)
{

    // get data from the map
    map_type::iterator it;
    {
        // Lock
        lock_type::scoped_lock l(lock);

        it = map.find(event);
    }

    // Make sure the data actually exists
    HPX_ASSERT(it != map.end());
    
    // Send the data to the client
    it->second.send_to_client(client_event);    

}

void
data_map_entry::send_to_client(const hpx::naming::id_type& client_event)
{
    // no synchronization necessary, should only get called once
    // (at least the client event has to make sure this only gets called once)
    send_callback(client_event);
}


