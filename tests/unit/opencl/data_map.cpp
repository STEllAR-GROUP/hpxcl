// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "cl_tests.hpp"

#include "../../../opencl/server/util/data_map.hpp"

static void cl_test( hpx::opencl::device local_device,
                     hpx::opencl::device cldevice )
{
    typedef hpx::serialization::serialize_buffer<char> buffer_type;

    // Create a data_map
    hpx::opencl::server::util::data_map map;

    // Create a promise
    hpx::promise<buffer_type> promise;

    // Create a future
    auto future = promise.get_future();

    // Create a cl_event
    cl_event event = (cl_event)5;

    // Make sure the promise did not get triggered yet
    hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
    HPX_TEST(!future.is_ready());

    {
        // Create some data
        buffer_type buffer("Test", sizeof("Test"), buffer_type::init_mode::copy);

        // Make sure the data isn't registered yet
        HPX_TEST(!map.has_data(event));

        // Register buffer in map
        map.add(event, buffer);

        // Make sure the data is now registered
        HPX_TEST(map.has_data(event));

        // Deallocate the buffer(out of scope).
        // Should get kept alive by the map.
    }

    // Make sure the promise did not get triggered yet
    hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
    HPX_TEST(!future.is_ready());

    // Trigger the promise
    map.get(event).send_data_to_client(promise.get_id());

    // Make sure the promise got triggered
    hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
    HPX_TEST(future.is_ready());

    // Make sure the data is correct
    auto data = future.get();
    HPX_TEST( strcmp(data.data(), "Test") == 0 );

    // Take the data out of the map
    map.remove(event);

}


