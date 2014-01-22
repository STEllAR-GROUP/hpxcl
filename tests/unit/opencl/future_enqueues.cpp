// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cl_tests.hpp"

#include <hpx/lcos/local/event.hpp>
#include <vector>


/*
 * This file tests the future<event> overloads of enqueue-calls. 
 *
 * This is done by asynchronously calculating (A+B)*(C-D),
 * Dependency tree:
 *
 * write_A -|
 *          |-> X = A+B -|
 * write_B -|            |
 *                       |-> Z = X * Y -> read_Z
 * write_C -|            |
 *          |-> Y = C-D -|
 * write_D -|
 */

static const int inputA[] = {  1,  5,  2,  4,  3};
static const int inputB[] = {  3,  7,  4,  1,  4};
static const int inputC[] = {  4,  9,  4,  7,  8};
static const int inputD[] = {  6,  3,  7,  2,  1};
static const int refX[]   = {  4, 12,  6,  5,  7};
static const int refY[]   = { -2,  6, -3,  5,  7};
static const int refZ[]   = { -8, 72,-18, 25, 49}; 
static const size_t DATASIZE = 5 * sizeof(int);

static const char gpu_prog[] =
"                                                                   \n"
"   __kernel void add(__global int* out,__global int* in1,          \n"
"                                       __global int* in2)          \n"
"   {                                                               \n"
"       size_t tid = get_global_id(0);                              \n"
"       out[tid] = in1[tid] + in2[tid];                             \n"
"   }                                                               \n"
"                                                                   \n"
"   __kernel void sub(__global int* out,__global int* in1,          \n"
"                                       __global int* in2)          \n"
"   {                                                               \n"
"       size_t tid = get_global_id(0);                              \n"
"       out[tid] = in1[tid] - in2[tid];                             \n"
"   }                                                               \n"
"                                                                   \n"
"   __kernel void mul(__global int* out,__global int* in1,          \n"
"                                       __global int* in2)          \n"
"   {                                                               \n"
"       size_t tid = get_global_id(0);                              \n"
"       out[tid] = in1[tid] * in2[tid];                             \n"
"   }                                                               \n"
"                                                                   \n";

static hpx::opencl::event generate_event(hpx::opencl::device cldevice,
                                         hpx::lcos::local::event *trigger)
{
    // create an event
    hpx::opencl::event event = cldevice.create_user_event().get();

    // wait for the trigger to trigger
    trigger->wait();

    // activate the event
    event.trigger();

    // return the event
    return event;
}



static void cl_test(hpx::opencl::device cldevice)
{
    // Make your life easier.
    typedef hpx::lcos::shared_future<hpx::opencl::event> future_event;

    // Generate kernels
    hpx::opencl::program prog = cldevice.create_program_with_source(gpu_prog);
    prog.build();
    hpx::opencl::kernel add_kernel = prog.create_kernel("add");
    hpx::opencl::kernel sub_kernel = prog.create_kernel("sub");
    hpx::opencl::kernel mul_kernel = prog.create_kernel("mul");

    // This trigger will start the entire process.
    hpx::lcos::local::event trigger;

    // This is the initial event that will get triggered by trigger
    future_event startEvent = 
            hpx::async(hpx::util::bind(generate_event, cldevice, &trigger));

    // Load buffers
    hpx::opencl::buffer bufA = cldevice.create_buffer(CL_MEM_READ_WRITE,
                                                            DATASIZE);
    hpx::opencl::buffer bufB = cldevice.create_buffer(CL_MEM_READ_WRITE,
                                                            DATASIZE);
    hpx::opencl::buffer bufC = cldevice.create_buffer(CL_MEM_READ_WRITE,
                                                            DATASIZE);
    hpx::opencl::buffer bufD = cldevice.create_buffer(CL_MEM_READ_WRITE,
                                                            DATASIZE);
    hpx::opencl::buffer bufX = cldevice.create_buffer(CL_MEM_READ_WRITE,
                                                            DATASIZE);
    hpx::opencl::buffer bufY = cldevice.create_buffer(CL_MEM_READ_WRITE,
                                                            DATASIZE);
    hpx::opencl::buffer bufZ = cldevice.create_buffer(CL_MEM_READ_WRITE,
                                                            DATASIZE);

    // Write to buffers
    future_event initAevent = bufA.enqueue_write(0, DATASIZE, inputA,
                                                                    startEvent);
    future_event initBevent = bufA.enqueue_write(0, DATASIZE, inputB,
                                                                    startEvent);
    future_event initCevent = bufA.enqueue_write(0, DATASIZE, inputC,
                                                                    startEvent);
    future_event initDevent = bufA.enqueue_write(0, DATASIZE, inputD,
                                                                    startEvent);
    
    // Create dependency lists
    std::vector<future_event> initABevent(2);
    initABevent.push_back(initAevent);
    initABevent.push_back(initBevent);
    std::vector<future_event> initCDevent(2);
    initCDevent.push_back(initCevent);
    initCDevent.push_back(initDevent);

    // set kernel args
    add_kernel.set_arg(0, bufX);
    add_kernel.set_arg(1, bufA);
    add_kernel.set_arg(2, bufB);

    sub_kernel.set_arg(0, bufY);
    sub_kernel.set_arg(1, bufC);
    sub_kernel.set_arg(2, bufD);
    
    mul_kernel.set_arg(0, bufZ);
    mul_kernel.set_arg(1, bufX);
    mul_kernel.set_arg(2, bufY);

    // set up work size
    hpx::opencl::work_size<1> dim;
    dim[0].offset = 0;
    dim[0].size = 5;

    // run add and sub kernels
    future_event add_event = add_kernel.enqueue(dim, initABevent);
    future_event sub_event = sub_kernel.enqueue(dim, initCDevent);

    // combine events
    std::vector<future_event> add_sub_event(2);
    add_sub_event.push_back(add_event);
    add_sub_event.push_back(sub_event);

    // run mul kernel
    future_event mul_event = mul_kernel.enqueue(dim, add_sub_event);

    // run read from result buffer
    future_event readZevent = bufZ.enqueue_read(0, DATASIZE, mul_event);

    // so far, nothing actually happened. everything is waiting for the trigger.
    // so: pull the trigger.
    trigger.set();
    
    // query data from the event
    boost::shared_ptr<std::vector<char>> chardata = 
                                    readZevent.get().get_data().get();

    // cast to int
    int* data = (int*)&(*chardata)[0];

    for(size_t i = 0; i < 5; i++)
    {
        std::cout << data[i] << std::endl;
    }

    

}


