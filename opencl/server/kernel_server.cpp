// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The Header of this class
#include "kernel.hpp"

// HPXCL tools
#include "../tools.hpp"

// other hpxcl dependencies
#include "device.hpp"
#include "buffer.hpp"

// HPX dependencies
#include <hpx/include/thread_executors.hpp>
#include <hpx/parallel/executors/service_executors.hpp>


using hpx::opencl::server::kernel;


// Constructor
kernel::kernel()
{}

// External destructor.
// This is needed because OpenCL calls only run properly on large stack size.
static void kernel_cleanup(uintptr_t kernel_id_ptr)
{

    cl_int err;

    HPX_ASSERT(hpx::opencl::tools::runs_on_medium_stack());

    cl_kernel kernel_id = reinterpret_cast<cl_kernel>(kernel_id_ptr);

    // Release the device memory
    if(kernel_id)
    {
        err = clReleaseKernel(kernel_id);
        cl_ensure_nothrow(err, "clReleaseKernel()");
    }
}

// Destructor
kernel::~kernel()
{

    hpx::threads::executors::default_executor exec(
                                          hpx::threads::thread_priority_normal,
                                          hpx::threads::thread_stacksize_medium);

    // run dectructor in a thread, as we need it to run on a large stack size
    hpx::threads::async_execute( exec, &kernel_cleanup, reinterpret_cast<uintptr_t>(kernel_id))
                                                                        .wait();


}

hpx::naming::id_type kernel::get_parent_device_id()
{
    return parent_device_id;
}

void
kernel::init( hpx::naming::id_type device_id, cl_program program,
              std::string kernel_name )
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_medium_stack());

    this->parent_device_id = std::move(device_id);
    this->parent_device = hpx::get_ptr
                          <hpx::opencl::server::device>(parent_device_id).get();
    this->kernel_id = NULL;

    // The opencl error variable
    cl_int err;

    // Create the cl_program
    kernel_id = clCreateKernel( program, kernel_name.c_str(), &err );
    cl_ensure(err, "clCreateKernel()");

}


void
kernel::set_arg(cl_uint arg_index, hpx::naming::id_type buffer_id)
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_medium_stack());
    cl_int err;

    // Get direct pointer to buffer
    auto buffer = hpx::get_ptr<hpx::opencl::server::buffer>(buffer_id).get();

    // Get cl_mem
    cl_mem mem_id = buffer->get_cl_mem();

    // Set the argument
    err = clSetKernelArg(kernel_id, arg_index, sizeof(cl_mem), &mem_id);
    cl_ensure(err, "clSetKernelArg()");

}

void
kernel::enqueue( hpx::naming::id_type && event_gid,
                 std::vector<std::size_t> size_vec,
                 std::vector<hpx::naming::id_type> && dependencies )
{
    HPX_ASSERT(hpx::opencl::tools::runs_on_medium_stack());

    cl_int err;
    cl_event return_event;

    // retrieve the dependency cl_events
    util::event_dependencies events( dependencies, parent_device.get() );

    // retrieve the command queue
    cl_command_queue command_queue = parent_device->get_kernel_command_queue();

    // prepare args for OpenCL call
    HPX_ASSERT( size_vec.size() % 3 == 0 );
    std::size_t size = size_vec.size() / 3;
    std::size_t* global_work_offset = size_vec.data() + 0 * size;
    std::size_t* global_work_size   = size_vec.data() + 1 * size;
    std::size_t* local_work_size    = size_vec.data() + 2 * size;

    // If local_work_size is not specified, let the OpenCL driver decide
    if(local_work_size[0] == 0){
        local_work_size = NULL;
    }

    // run the OpenCL-call
    err = clEnqueueNDRangeKernel( command_queue, kernel_id,
                                  static_cast<cl_uint>(size),
                                  global_work_offset,
                                  global_work_size,
                                  local_work_size,
                                  static_cast<cl_uint>(events.size()),
                                  events.get_cl_events(),
                                  &return_event );
    cl_ensure(err, "clEnqueueNDRangeKernel()");

    // register the cl_event to the client event
    parent_device->register_event(event_gid, return_event);

}

