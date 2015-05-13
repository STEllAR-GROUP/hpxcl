// Copyright (c)       2015 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "cl_tests.hpp"

#include "../../../opencl/util/enqueue_overloads.hpp"
#include "../../../opencl/lcos/event.hpp"

class test_client{
    public:
    HPX_OPENCL_GENERATE_ENQUEUE_OVERLOADS(int, func, int, int);
};


hpx::future<int>
test_client::func_impl(int a, int b, std::vector<hpx::naming::id_type> ids){
    return hpx::make_ready_future<int>(ids.size() + 1000 * a + 100 * b);
};




static void cl_test(hpx::opencl::device cldevice){

    hpx::opencl::lcos::event<void> event;

    hpx::shared_future<void> sfut = event.get_future();
    std::vector<hpx::shared_future<void>> vsfut1 = {sfut};
    std::vector<hpx::shared_future<void>> vsfut2 = {sfut, sfut};

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

