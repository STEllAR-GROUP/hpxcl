// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "kernel.hpp"
#include "program.hpp"
#include "../tools.hpp"

#include <string>
#include <sstream>

#include <CL/cl.h>


using namespace hpx::opencl::server;

CL_FORBID_EMPTY_CONSTRUCTOR(kernel);


kernel::kernel(hpx::naming::id_type program_id, std::string kernel_name)
{
    this->kernel_id = NULL;
    this->parent_program_id = program_id;
    this->parent_program = hpx::get_ptr
                         <hpx::opencl::server::program>(parent_program_id).get();
    this->parent_device_id = parent_program->get_device_id();
    this->parent_device = hpx::get_ptr
                          <hpx::opencl::server::device>(parent_device_id).get();

    // initialize the cl_program object
    cl_int err;
    kernel_id = clCreateKernel(parent_program->get_cl_program(),
                               kernel_name.c_str(), &err);
    clEnsure(err, "clCreateKernel()");
                              
}

kernel::~kernel()
{
    cl_int err;

    // release the cl_program object
    if(kernel_id)
    {
        err = clReleaseKernel(kernel_id);
        clEnsure_nothrow(err, "clReleaseKernel()");
        kernel_id = NULL;
    }

}

void
kernel::set_arg(cl_uint arg_index, hpx::opencl::buffer arg)
{
    
    // Get local pointer to buffer
    boost::shared_ptr<hpx::opencl::server::buffer>
    buffer_local = hpx::get_ptr<hpx::opencl::server::buffer>(arg.get_gid())
                                                                         .get();

    // Get cl_mem
    cl_mem mem_id = buffer_local->get_cl_mem();

    // Set the argument
    cl_int err;
    err = clSetKernelArg(kernel_id, arg_index, sizeof(cl_mem), &mem_id);
    clEnsure(err, "clSetKernelArg()");

}

hpx::opencl::event
kernel::enqueue(cl_uint work_dim, std::vector<std::vector<size_t>> args,
                                  std::vector<hpx::opencl::event> events)
{

    // Ensure correctness of input data
    BOOST_ASSERT(args.size() == 3);

    // Fetch command queue
    cl_command_queue command_queue = parent_device->get_work_command_queue();
    
    // Convert vectors to pointers
    size_t* global_work_offset = NULL;
    size_t* global_work_size   = NULL;
    size_t* local_work_size    = NULL;
    if(args[0].size() == work_dim) global_work_offset = &(args[0][0]); 
    if(args[1].size() == work_dim) global_work_size   = &(args[1][0]); 
    if(args[2].size() == work_dim) local_work_size    = &(args[2][0]); 

    // Get the cl_event dependency list
    std::vector<cl_event> cl_events_list = hpx::opencl::event::
                                                    get_cl_events(events);
    cl_event* cl_events_list_ptr = NULL;
    if(!cl_events_list.empty())
    {
        cl_events_list_ptr = &cl_events_list[0];
    }

    // Enqueue the kernel
    cl_int err;
    cl_event returnEvent;
    err = clEnqueueNDRangeKernel(command_queue, kernel_id, work_dim,
                                 global_work_offset,
                                 global_work_size,
                                 local_work_size,
                                 (cl_uint)events.size(),
                                 cl_events_list_ptr,
                                 &returnEvent);
    clEnsure(err, "clEnqueueNDRangeKernel()");

    // Return the event
    return hpx::opencl::event(
               hpx::components::new_<hpx::opencl::server::event>(
                                hpx::find_here(),
                                parent_device_id,
                                (clx_event) returnEvent
                            ));

}





