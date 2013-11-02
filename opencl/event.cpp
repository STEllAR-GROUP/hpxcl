// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/lcos/future.hpp>

#include "event.hpp"


using hpx::opencl::event;


cl_event
event::get_cl_events(hpx::opencl::event event){
    // TODO implement faster version

    std::vector<hpx::opencl::event> vector(1);
    vector.push_back(event);
    return get_cl_events(vector)[0];

}

std::vector<cl_event>
event::get_cl_events(std::vector<hpx::opencl::event> events)
{

    // Step 1: Fetch opencl event component pointers
    std::vector<hpx::lcos::future<boost::shared_ptr
            <hpx::opencl::server::event>>> event_server_futures(events.size());
    BOOST_FOREACH(hpx::opencl::event & event, events)
    {
        BOOST_ASSERT(event.get_gid());
        event_server_futures.push_back(
            hpx::get_ptr<hpx::opencl::server::event>(event.get_gid()));   
    }

    // Wait for Step 1 to finish
    std::vector<boost::shared_ptr<hpx::opencl::server::event>>
            event_servers(events.size());
    BOOST_FOREACH(hpx::lcos::future<boost::shared_ptr
                    <hpx::opencl::server::event>> & event_server_future, 
                            event_server_futures)
    {
        event_servers.push_back(event_server_future.get());
    }

    // Fetch the cl_event pointers from event servers and create eventlist
    std::vector<cl_event> cl_events_list(events.size());
    BOOST_FOREACH(boost::shared_ptr<hpx::opencl::server::event> & event_server,
                                                                event_servers)
    {
        cl_events_list.push_back(event_server->get_cl_event());
    }  

    return cl_events_list;

}

