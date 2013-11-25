// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime.hpp>
#include <hpx/lcos/local/event.hpp>

namespace hpx { namespace opencl { namespace server {

// This function triggers an hpx::lcos::local::event from an external thread
void trigger_event_from_external(hpx::runtime * rt,
                                 hpx::lcos::local::event * event);


}}}
