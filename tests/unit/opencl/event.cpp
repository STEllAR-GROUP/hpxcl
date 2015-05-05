// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cl_tests.hpp"

#include "../../../opencl/lcos/event.hpp"

static void cl_test(hpx::opencl::device cldevice)
{
    hpx::opencl::lcos::event<void> event;

    hpx::future<void> fut = event.get_future();
}


