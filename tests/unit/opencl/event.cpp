// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>
#include <hpx/hpx_main.hpp>

#include <hpx/include/iostreams.hpp>

#include <iomanip>

void dummy(){}


int main()
{
    typedef hpx::lcos::promise<void> promise_type;
    typedef typename promise_type::wrapped_type shared_state_type;


    promise_type promise;
    auto gid = promise.get_gid();


    auto future = promise.get_future();
    auto future_data = hpx::lcos::detail::get_shared_state(future);
    auto shared_state = boost::static_pointer_cast<shared_state_type>(future_data);


    auto gid2 = shared_state->get_gid();

    hpx::cout << gid << hpx::endl;
    hpx::cout << gid2 << hpx::endl;
    hpx::cout << std::hex << (gid.get_msb() & 0x0000000000FFFFFF) << hpx::endl;
    hpx::cout << std::hex << (gid2.get_msb() & 0x0000000000FFFFFF) << hpx::endl;


    return 0;
}


