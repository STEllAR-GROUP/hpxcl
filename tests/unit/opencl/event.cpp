// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>
#include <hpx/hpx_main.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <hpx/include/iostreams.hpp>

void dummy(){}


int hpx_main()
{

    {
        typedef hpx::lcos::promise<void> promise_type;
        typedef typename promise_type::wrapped_type shared_state_type;


        promise_type promise;


        auto future = promise.get_future();
        auto future_data = hpx::lcos::detail::get_shared_state(future);
        auto shared_state = boost::static_pointer_cast<shared_state_type>(future_data);


        auto gid2 = shared_state->get_gid();
        auto gid = promise.get_gid();
        

        HPX_TEST_EQ(gid, gid2);

        hpx::cout << gid << hpx::endl;
        hpx::cout << gid2 << hpx::endl;
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    return hpx::init(argc, argv);
}
