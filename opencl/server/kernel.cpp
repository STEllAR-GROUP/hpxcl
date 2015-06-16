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
#include "util/hpx_cl_interop.hpp"

// HPX dependencies
#include <hpx/include/thread_executors.hpp>


using hpx::opencl::server::kernel;


// Constructor
kernel::kernel()
{}

// External destructor.
// This is needed because OpenCL calls only run properly on large stack size.
static void kernel_cleanup(uintptr_t kernel_id_ptr)
{

    cl_int err;

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

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
                                          hpx::threads::thread_stacksize_large);

    // run dectructor in a thread, as we need it to run on a large stack size
    hpx::async( exec, &kernel_cleanup, reinterpret_cast<uintptr_t>(kernel_id))
                                                                        .wait();


}


void
kernel::init( hpx::naming::id_type device_id, cl_program program,
              std::string kernel_name )
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

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

