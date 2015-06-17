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


data_map_entry
data_map::get(cl_event event)
{
    data_map_entry result;

    // get data from the map
    {
        // Lock
        lock_type::scoped_lock l(lock);

        // Retrieve the data from the map
        map_type::iterator it = map.find(event);
    
        // Make sure the data actually exists
        HPX_ASSERT(it != map.end());
    
        // Get the data entry
        result = it->second; 
    }

    return result;
}

bool
data_map::has_data(cl_event event)
{

    bool result = true;
    {
        // Lock
        lock_type::scoped_lock l(lock);

        // Try to find the entry
        map_type::iterator it = map.find(event);

        // Check wether or not we found the entry
        if(it == map.end())
            result = false;
    }

    return result;
}

void
data_map_entry::send_data_to_client(const hpx::naming::id_type& client_event)
{
    // no synchronization necessary, should only get called once
    // (at least the client event has to make sure this only gets called once)
    send_callback(client_event);
}


