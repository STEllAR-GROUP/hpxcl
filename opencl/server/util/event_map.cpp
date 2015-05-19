// Copyright (c)    2015 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The Header of this class
#include "event_map.hpp"


using hpx::opencl::server::util::event_map;

event_map::gid_entry::gid_entry(boost::uint64_t msb_, boost::uint64_t lsb_){
    msb = msb_;
    lsb = lsb_;
}

event_map::gid_entry::gid_entry(const hpx::naming::id_type & gid){
    msb = gid.get_msb();
    lsb = gid.get_lsb();
}

event_map::event_map(){

}

event_map::~event_map(){
    // Correct use removes all entries from this map before deletion
    HPX_ASSERT(events.empty());
    HPX_ASSERT(waits.empty());
}


void
event_map::add(const hpx::naming::id_type & gid, cl_event event){
    gid_entry key(gid);

    std::cout << key.msb << " - " << key.lsb << std::endl;
    {
        // Lock
        lock_type::scoped_lock l(lock);
        
        // Insert
        events.insert(map_type::value_type(key, event));
        HPX_ASSERT(events.at(key) == event);

        // Retrieve end delete condition variable if exists
        waitmap_type::iterator it = waits.find(key);
        if(it != waits.end()){
            // Notify waiting threads
            it->second->notify_all();
            waits.erase(it);
        }
    }

}

cl_event
event_map::get(const hpx::naming::id_type& gid){
    gid_entry key(gid);

    map_type::iterator it;
    {
        // Lock
        lock_type::scoped_lock l(lock);
        
        // Try to retrieve
        it = events.find(key);
    }

    // On success, return
    if(it != events.end()){
        return it->second;
    }

    // On failure, try again and register callback
    waitmap_type::value_type waits_entry(key, std::make_shared<condition_type>());
    {
        // Lock
        lock_type::scoped_lock l(lock);
        
        // Try to retrieve
        it = events.find(key);
    
        // On success, return
        if(it != events.end()){
            return it->second;
        }

        // On failure, register condition variable (or retrieve existing one)
        auto inserted_condvar = waits.insert(std::move(waits_entry));
        
        // Unwrap the condition variable
        std::shared_ptr<condition_type> condition
            = inserted_condvar.first->second;

        // Wait for some other thread to add() the missing key
        condition->wait(l);

        // This should now definitely return the requested item.
        it = events.find(key);
        HPX_ASSERT(it != events.end());

        return it->second;

    }

}

void 
event_map::remove(boost::uint64_t gid_msb, boost::uint64_t gid_lsb)
{
    gid_entry key(gid_msb, gid_lsb);

    cl_event event;
    {
        // Lock
        lock_type::scoped_lock l(lock);
    
        // Find Element
        std::cout << key.msb << " - " << key.lsb << std::endl;
        auto it = events.find(key);
        HPX_ASSERT(it != events.end());

        // Unwrap event
        event = it->second;
    }

    // run deletion callback
    deletion_callback(event);

    {
        // Lock
        lock_type::scoped_lock l(lock);
    
        // Remove element
        events.erase(key);
    }
}

void
event_map::register_deletion_callback(std::function<void(cl_event)> && callback){
    this->deletion_callback = callback;
}


