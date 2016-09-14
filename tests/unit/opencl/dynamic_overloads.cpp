// Copyright (c)       2015 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "cl_tests.hpp"

#include "../../../opencl/util/enqueue_overloads.hpp"
#include "../../../opencl/lcos/event.hpp"

#include "register_event.hpp"

class test_client{
    public:
    template<typename ...Deps>
    hpx::future<int> func(int a, int b, Deps &&... dependencies )
    {
        // combine dependency futures in one std::vector
        using hpx::opencl::util::enqueue_overloads::resolver;
        //auto deps = resolver(std::forward<Deps>(dependencies)...);

        //return func_impl( std::move(a), std::move(b), std::move(deps) );
    }

    hpx::future<int> func_impl( int && a, int && b,
                                hpx::opencl::util::resolved_events && ids );
};


hpx::future<int>
test_client::func_impl( int && a, int && b,
                        hpx::opencl::util::resolved_events && ids){
    return hpx::make_ready_future<int>(ids.event_ids.size() + 1000 * a + 100 * b);
};




static void cl_test( hpx::opencl::device local_device, 
                     hpx::opencl::device cldevice ){

    hpx::opencl::lcos::event<void> event(local_device.get_id());
    register_event(local_device, event.get_event_id());

    hpx::shared_future<void> sfut = event.get_future();
    std::vector<hpx::shared_future<void>> vsfut1;
    vsfut1.push_back( sfut );
    std::vector<hpx::shared_future<void>> vsfut2;
    vsfut2.push_back( sfut );
    vsfut2.push_back( sfut );

    test_client t;

    HPX_TEST_EQ( 5300, t.func(5, 3                               ).get() );
    HPX_TEST_EQ( 5301, t.func(5, 3, sfut                         ).get() );
    HPX_TEST_EQ( 5302, t.func(5, 3, sfut, sfut                   ).get() );
    HPX_TEST_EQ( 5301, t.func(5, 3, vsfut1                       ).get() );
    HPX_TEST_EQ( 5302, t.func(5, 3, vsfut2                       ).get() );
    HPX_TEST_EQ( 5302, t.func(5, 3, vsfut1, vsfut1               ).get() );
    HPX_TEST_EQ( 5303, t.func(5, 3, vsfut1, vsfut2               ).get() );
    HPX_TEST_EQ( 5303, t.func(5, 3, vsfut2, vsfut1               ).get() );
    HPX_TEST_EQ( 5304, t.func(5, 3, vsfut2, vsfut2               ).get() );
    HPX_TEST_EQ( 5304, t.func(5, 3, sfut, sfut, vsfut2           ).get() );
    HPX_TEST_EQ( 5304, t.func(5, 3, sfut, vsfut2, sfut           ).get() );
    HPX_TEST_EQ( 5304, t.func(5, 3, vsfut2, sfut, sfut           ).get() );
    HPX_TEST_EQ( 5305, t.func(5, 3, sfut, vsfut2, vsfut2         ).get() );
    HPX_TEST_EQ( 5305, t.func(5, 3, vsfut2, sfut, vsfut2         ).get() );
    HPX_TEST_EQ( 5305, t.func(5, 3, vsfut2, vsfut2, sfut         ).get() );


};

