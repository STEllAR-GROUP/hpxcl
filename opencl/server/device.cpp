// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The Header of this class
#include "device.hpp"

// HPXCL tools
#include "../tools.hpp"

// other hpxcl dependencies
#include "buffer.hpp"
#include "program.hpp"

// HPX dependencies
#include <hpx/include/thread_executors.hpp>

using namespace hpx::opencl::server;


// Constructor
device::device()
{
    // Register the event deletion callback function at the event map
    event_map.register_deletion_callback(&delete_event);
}

// External destructor.
// This is needed because OpenCL calls only run properly on large stack size.
static void device_cleanup(uintptr_t command_queue_ptr,
                           uintptr_t context_ptr)
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack());

    cl_int err;

    cl_command_queue command_queue = reinterpret_cast<cl_command_queue>(command_queue_ptr);
    cl_context context = reinterpret_cast<cl_context>(context_ptr);

    // Release command queue
    if(command_queue)
    {
        err = clFinish(command_queue);
        cl_ensure_nothrow(err, "clFinish()");
        err = clReleaseCommandQueue(command_queue);
        cl_ensure_nothrow(err, "clReleaseCommandQueue()");
        command_queue = NULL;
    }

    // Release context
    if(context)
    {
        err = clReleaseContext(context);
        cl_ensure_nothrow(err, "clReleaseContext()");
        context = NULL;
    }

}

// Destructor
device::~device()
{

    hpx::threads::executors::default_executor exec(
                                          hpx::threads::thread_priority_normal,
                                          hpx::threads::thread_stacksize_large);

    // run dectructor in a thread, as we need it to run on a large stack size
    hpx::async( exec, &device_cleanup, (uintptr_t)command_queue,
                                       (uintptr_t)context).wait();

}

// Initialization function.
// Needed because cl_device_id can not be serialized.
void
device::init(cl_device_id _device_id, bool enable_profiling)
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack());

    this->device_id = _device_id;

    cl_int err;

    // Retrieve platformID
    err = clGetDeviceInfo(this->device_id, CL_DEVICE_PLATFORM,
                          sizeof(platform_id), &platform_id, NULL);
    cl_ensure(err, "clGetDeviceInfo()");

    // Create Context
    cl_context_properties context_properties[] =
                        {CL_CONTEXT_PLATFORM,
                         (cl_context_properties) platform_id,
                         0};
    context = clCreateContext(context_properties,
                              1,
                              &this->device_id,
                              error_callback,
                              this,
                              &err);
    cl_ensure(err, "clCreateContext()");

    // Get supported device queue properties
    hpx::serialization::serialize_buffer<char> supported_queue_properties_data =
                                    get_device_info(CL_DEVICE_QUEUE_PROPERTIES);
    cl_command_queue_properties supported_queue_properties =
                *( reinterpret_cast<cl_command_queue_properties *>(
                                       supported_queue_properties_data.data()));

    // Initialize command queue properties
    cl_command_queue_properties command_queue_properties = 0;

    // If supported, add OUT_OF_ORDER_EXEC_MODE
    if(supported_queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
        command_queue_properties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;

    // If supported and wanted, add PROFILING
    if(enable_profiling &&
                       (supported_queue_properties & CL_QUEUE_PROFILING_ENABLE))
        command_queue_properties |= CL_QUEUE_PROFILING_ENABLE;

    // Create Command Queue
    #ifdef CL_VERSION_2_0
        cl_queue_properties queue_properties[] = {CL_QUEUE_PROPERTIES,
                              (cl_queue_properties) command_queue_properties,
                              (cl_queue_properties) 0};
        command_queue = clCreateCommandQueueWithProperties(context, device_id,
                                                        queue_properties, &err);
        cl_ensure(err, "clCreateCommandQueueWithProperties()");
    #else
        command_queue = clCreateCommandQueue(context, device_id,
                                             command_queue_properties, &err);
        cl_ensure(err, "clCreateCommandQueue()");
    #endif
}


hpx::serialization::serialize_buffer<char>
device::get_device_info(cl_device_info info_type)
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack());

    // Declairing the cl error code variable
    cl_int err;

    // Query for size
    std::size_t param_size;
    err = clGetDeviceInfo(device_id, info_type, 0, NULL, &param_size);
    cl_ensure(err, "clGetDeviceInfo()");

    // Retrieve
    hpx::serialization::serialize_buffer<char> info( new char[param_size],
                                            param_size,
                                            hpx::serialization::serialize_buffer<char>::take);
    err = clGetDeviceInfo(device_id, info_type, param_size, info.data(), 0);
    cl_ensure(err, "clGetDeviceInfo()");

    // Return
    return info;

}


hpx::serialization::serialize_buffer<char>
device::get_platform_info(cl_platform_info info_type)
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack());

    // Declairing the cl error code variable
    cl_int err;

    // Query for size
    std::size_t param_size;
    err = clGetPlatformInfo(platform_id, info_type, 0, NULL, &param_size);
    cl_ensure(err, "clGetPlatformInfo()");

    // Retrieve
    hpx::serialization::serialize_buffer<char>
    info( new char[param_size], param_size,
          hpx::serialization::serialize_buffer<char>::take);
    err = clGetPlatformInfo(platform_id, info_type, param_size, info.data(), 0);
    cl_ensure(err, "clGetPlatformInfo()");

    // Return
    return info;

}


void CL_CALLBACK
device::error_callback(const char* errinfo, const void* info, std::size_t info_size,
                                                void* _thisp)
{
    device* thisp = (device*) _thisp;
    hpx::cerr << "device(" << thisp->device_id << "): CONTEXT_ERROR: "
             << errinfo << hpx::endl;
}

cl_context
device::get_context()
{
    return context;
}

cl_device_id
device::get_device_id()
{
    return device_id;
}

hpx::id_type
device::create_buffer( cl_mem_flags flags, std::size_t size )
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack());

    // Create new buffer
    hpx::id_type buf = hpx::components::new_<hpx::opencl::server::buffer>
                                                     ( hpx::find_here() ).get();

    // Initialize buffer locally
    std::shared_ptr<hpx::opencl::server::buffer> buffer_server =
                        hpx::get_ptr<hpx::opencl::server::buffer>( buf ).get();

    buffer_server->init(get_id(), flags, size);

    return buf;
}

hpx::id_type
device::create_program_with_source(
    hpx::serialization::serialize_buffer<char> src )
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack());

    // Create new program
    hpx::id_type prog = hpx::components::new_<hpx::opencl::server::program>
                                                     ( hpx::find_here() ).get();

    // Initialize buffer locally
    std::shared_ptr<hpx::opencl::server::program> program_server =
                        hpx::get_ptr<hpx::opencl::server::program>( prog ).get();

    program_server->init_with_source( get_id(), src );

    return prog;
}

hpx::id_type
device::create_program_with_binary(
    hpx::serialization::serialize_buffer<char> binary )
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack());

    // Create new program
    hpx::id_type prog = hpx::components::new_<hpx::opencl::server::program>
                                                     ( hpx::find_here() ).get();

    // Initialize buffer locally
    std::shared_ptr<hpx::opencl::server::program> program_server =
                        hpx::get_ptr<hpx::opencl::server::program>( prog ).get();

    program_server->init_with_binary( get_id(), binary );

    return prog;
}

void
device::release_event(hpx::naming::gid_type gid)
{
    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack());

    // release data registered on event
    delete_event_data(event_map.get(gid));

    // delete event from map
    event_map.remove(gid);

}


void
device::delete_event( cl_event event )
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack());

    // delete the actual cl_event object
    cl_int err;
    err = clReleaseEvent(event);
    cl_ensure(err, "clReleaseEvent()");

}

void
device::register_event( const hpx::naming::id_type & gid, cl_event event )
{

    // Add pair to event_map
    event_map.add(gid, event);

}

cl_event
device::retrieve_event( const hpx::naming::id_type & gid )
{

    // Get event from event_map
    return event_map.get(gid);

}


cl_command_queue
device::get_read_command_queue()
{
    return command_queue;
}

cl_command_queue
device::get_write_command_queue()
{
    return command_queue;
}

cl_command_queue
device::get_kernel_command_queue()
{
    return command_queue;
}

void
device::wait_for_cl_event(cl_event event)
{
    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack());

    cl_int err;
    cl_int execution_state = CL_RUNNING;

    // Loop until the event state turns to true.
    // Previous attempts used clSetEventCallback, but it turned out to
    // be really slow.
    for(std::size_t i = 0; execution_state != CL_COMPLETE; i++){

        // Do exponential backup, stolen from the hpx spinlock class
        hpx::lcos::local::spinlock::yield(i);

        // Query OpenCL for the event state
        err = clGetEventInfo( event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                              sizeof(cl_int), &execution_state, NULL );
        cl_ensure(err, "clGetEventInfo");

    }

    // Check for internal errors
    cl_ensure(execution_state, "OpenCL Internal Error!");

}

void
device::delete_event_data(cl_event event)
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack());

    cl_int err;

    // return if no data is registered
    if(!event_data_map.has_data(event))
        return;

    // wait for event to trigger (clEnqueueX-call could still be using
    //                            the memory)
    wait_for_cl_event(event);

    // release the data
    event_data_map.remove(event);

}

void
device::activate_deferred_event(hpx::naming::id_type event_id)
{
    // get the cl_event
    cl_event event = event_map.get(event_id);

    // wait for the cl_event to complete
    wait_for_cl_event(event);

    // trigger the client event
    hpx::trigger_lco_event(event_id, false);

}

void
device::activate_deferred_event_with_data(hpx::naming::id_type event_id)
{

    // get the cl_event
    cl_event event = event_map.get(event_id);

    // find the data associated with the event
    auto data = event_data_map.get(event);

    // wait for the event to trigger
    wait_for_cl_event(event);

    // send the data to the client
    data.send_data_to_client(event_id);

}
