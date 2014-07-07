// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/apply.hpp>

#include "../../opencl.hpp"

using namespace hpx::opencl;

// Helper function to create an OpenCL device
static device create_cl_device()
{

    // Get list of available OpenCL Devices.
    std::vector<device> devices = get_devices( hpx::find_here(),
                                               CL_DEVICE_TYPE_ALL,
                                               "OpenCL 1.1" ).get();

    // Check whether there are any devices
    if(devices.size() < 1)
    {
        hpx::cerr << "No OpenCL devices found!" << hpx::endl;
        exit(hpx::finalize());
    }

    // Create a device component from the first device found
    return devices[0];

}

static void print(int id, const char* msg)
{
    hpx::cout << "[" << id << "] " << msg << hpx::endl;
}

// Counts down from 5, then triggers the event
void wait_and_trigger(event user_event, device cldevice)
{
    hpx::this_thread::sleep_for(boost::posix_time::milliseconds(100));
    print(1, "Triggering user_event in 5 ...");
    hpx::this_thread::sleep_for(boost::posix_time::milliseconds(1000));
    print(1, "                         4 ...");
    hpx::this_thread::sleep_for(boost::posix_time::milliseconds(1000));
    print(1, "                         3 ...");
    hpx::this_thread::sleep_for(boost::posix_time::milliseconds(1000));
    print(1, "                         2 ...");
    hpx::this_thread::sleep_for(boost::posix_time::milliseconds(1000));
    print(1, "                         1 ...");
    hpx::this_thread::sleep_for(boost::posix_time::milliseconds(1000));
    print(1, "Triggering user_event ...");
    user_event.trigger();
    print(1, "user_event triggered.");
}
HPX_PLAIN_ACTION(wait_and_trigger, wait_and_trigger_action);

// Waits for future to trigger
void wait_for_future(hpx::lcos::shared_future<void> future)
{
    print(2, "Waiting for user_event_future to trigger ...");
    future.get();
    print(2, "user_event_future triggered.");
}


// hpx_main, is the actual main called by hpx
int hpx_main(int argc, char* argv[])
{

    {
        // Create an OpenCL device
        print(0, "Creating device ... ");
        device cldevice = create_cl_device();
    
        // Create a user event
        print(0, "Creating user_event ...");
        event user_event = cldevice.create_user_event().get();
    
        // Create future from event
        print(0, "Creating user_event_future from user_event ...");
        hpx::lcos::shared_future<void> user_event_future =
                                                        user_event.get_future();
    
        // Create event from future
        print(0, "Creating user_event_future_event from user_event_future ...");
        event user_event_future_event = cldevice.create_future_event(
                                                               user_event_future
                                                                        ).get();
    
        // Run the wait_for_future function
        print(0, "Starting asynchronous functions ...");
        hpx::apply(&wait_for_future, user_event_future);
        typedef wait_and_trigger_action func2;
        hpx::apply<func2>(hpx::find_here(), user_event, cldevice);
    
        // Wait for user event
        print(0, "Waiting for user_event_future_event to trigger ...");
        user_event_future_event.await();
        print(0, "user_event_future_event triggered.");
    
    }

    print(0, "end of main program");

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


