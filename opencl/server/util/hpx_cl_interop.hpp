// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>
#include <hpx/runtime.hpp>

#include "../../cl_headers.hpp"

namespace hpx { namespace opencl { namespace server { namespace util {

// This function triggers an hpx::lcos::local::event from an external thread
void set_promise_from_external( hpx::runtime * rt,
                                hpx::lcos::local::promise<cl_int> * promise,
                                cl_int value );


}}}}
