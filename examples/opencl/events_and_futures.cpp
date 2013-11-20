// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>

#include "../../opencl/std.hpp"
#include "../../opencl/device.hpp"

using namespace hpx::opencl;

// Helper function to create an OpenCL device
static device create_cl_device()
{

    // Get list of available OpenCL Devices.
    std::vector<clx_device_id> devices = get_device_ids( hpx::find_here(),
                                                         CL_DEVICE_TYPE_ALL,
                                                         1.1f ).get();

    // Check whether there are any devices
    if(devices.size() < 1)
    {
        hpx::cerr << "No OpenCL devices found!" << hpx::endl;
        exit(hpx::finalize());
    }

    // Create a device component from the first device found
    return device(
             hpx::components::new_<server::device>(hpx::find_here(), devices[0])
                       );

}

static void print(int id, const char* msg)
{
    std::cout << "[" << id << "] " << msg << std::endl;
}

// Counts down from 5, then triggers the event
void wait_and_trigger(event user_event, device cldevice)
{
    sleep(1);
    print(1, "Triggering user_event in 5 ...");
    sleep(1);
    print(1, "                         4 ...");
    sleep(1);
    print(1, "                         3 ...");
    sleep(1);
    print(1, "                         2 ...");
    sleep(1);
    print(1, "                         1 ...");
    sleep(1);
    print(1, "Triggering user_event ...");
    cldevice.trigger_user_event(user_event);
}
HPX_PLAIN_ACTION(wait_and_trigger, wait_and_trigger_action);

// Waits for future to trigger
void wait_for_future(intptr_t future)
{
    print(2, "Waiting for user_event_future to trigger ...");
    ((hpx::lcos::future<void>*)future)->get();
    print(2, "user_event_future triggered.");
}
HPX_PLAIN_ACTION(wait_for_future, wait_for_future_action);


// hpx_main, is the actual main called by hpx
int hpx_main(int argc, char* argv[])
{
    // Create an OpenCL device
    print(0, "Creating device ... ");
    device cldevice = create_cl_device();

    // Create a user event
    print(0, "Creating user_event ...");
    event user_event = cldevice.create_user_event().get();

    // Create future from event
    print(0, "Creating user_event_future from user_event ...");
    hpx::lcos::future<void> user_event_future = user_event.get_future();

    // Create event from future
    print(0, "Creating user_event_future_event from user_event_future ...");
    event user_event_future_event = cldevice.create_future_event(
                                                           user_event_future
                                                                    ).get();

    // Run the wait_for_future function
    print(0, "Starting asynchronous functions ...");
    typedef wait_for_future_action func;
    hpx::apply<func>(hpx::find_here(), (intptr_t)&user_event_future);
    typedef wait_and_trigger_action func2;
    hpx::apply<func2>(hpx::find_here(), user_event, cldevice);

    // Wait for user event
    print(0, "Waiting for user_event_future_event to trigger ...");
    user_event_future_event.await();
    print(0, "user_event_future_event triggered.");

    // End the program
    return hpx::finalize();
}

// Main, initializes HPX
int main(int argc, char* argv[]){

    // initialize HPX, run hpx_main
    hpx::start(argc, argv);

    // wait for hpx::finalize being called
    return hpx::stop();
}


