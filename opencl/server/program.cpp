// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The Header of this class
#include "program.hpp"

// HPXCL tools
#include "../tools.hpp"

// other hpxcl dependencies
#include "device.hpp"

// HPX dependencies
#include <hpx/include/thread_executors.hpp>


using hpx::opencl::server::program;


// Constructor
program::program()
{}

// External destructor.
// This is needed because OpenCL calls only run properly on large stack size.
static void program_cleanup(uintptr_t program_cl_ptr)
{

    cl_int err;

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    cl_program program_cl = reinterpret_cast<cl_program>(program_cl_ptr);

    // Release the device memory
    if(program_cl)
    {
        err = clReleaseProgram(program_cl);
        cl_ensure_nothrow(err, "clReleaseProgram()");
    }
}

// Destructor
program::~program()
{

    hpx::threads::executors::default_executor exec(
                                          hpx::threads::thread_priority_normal,
                                          hpx::threads::thread_stacksize_large);

    // run dectructor in a thread, as we need it to run on a large stack size
    hpx::async( exec, &program_cleanup, reinterpret_cast<uintptr_t>(program_cl))
                                                                        .wait();


}


void
program::init_with_source( hpx::naming::id_type device_id, 
                           hpx::serialization::serialize_buffer<char> src )
{

    HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack()); 

    this->parent_device_id = device_id;
    this->parent_device = hpx::get_ptr
                          <hpx::opencl::server::device>(parent_device_id).get();
    this->program_cl = NULL;

    // Retrieve the context from parent class
    cl_context context = parent_device->get_context();

    // The opencl error variable
    cl_int err;

    // Set up data for OpenCL call
    HPX_ASSERT(src.size() > 0);
    std::size_t src_size = src.size();
    const char* src_data = src.data();
    if(src_data[src_size - 1] == '\0'){
        // Decrease one if zero-terminated, as
        // OpenCL specifies 'length of source string excluding null terminator'
        src_size --;
    }

    // Create the cl_program
    program_cl = clCreateProgramWithSource( context, 1, &src_data, &src_size,
                                            &err );
    cl_ensure(err, "clCreateProgramWithSource()");

}



