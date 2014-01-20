// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "future_execution.hpp"

#include <vector>


namespace hpx { namespace opencl {

    std::vector<event>
    wait_for_futures(const std::vector<hpx::lcos::unique_future<event>> &future_list){

        // Create list of events that will get filled by futures
        std::vector<event> event_list(future_list.size());

        // Wait for all futures
        for(size_t i = 0; i < future_list.size(); i++)
        {
            event_list[i] = future_list[i].get();
        }

        // Return the filled event list
        return event_list;
    }


}}


