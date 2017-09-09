// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cl_tests.hpp"


/*
 * This test is meant to verify the kernel creation and execution functionality.
 */

CREATE_BUFFER(invalid_program_src,
"                                                                          \n"
"   __kernel void hello_world(__global char * in, __global char * out)     \n"
"   {                                                                      \n"
"       size_t tid = get_global_id(0);                                     \n"
"       out[tid] = (char)(in[unknown_variable] + tid);                     \n"
"   }                                                                      \n"
"                                                                          \n");

CREATE_BUFFER(program_src,
"                                                                          \n"
"   __kernel void hello_world(__global char * in, __global char * out)     \n"
"   {                                                                      \n"
"       size_t tid = get_global_id(0);                                     \n"
"       out[tid] = (char)(in[tid] + tid);                                  \n"
"   }                                                                      \n"
"                                                                          \n");



#define DATASIZE (sizeof("Hello, World!"))

const char initdata_arr[] = { ('H'  - static_cast<char>( 0)),
                              ('e'  - static_cast<char>( 1)),
                              ('l'  - static_cast<char>( 2)),
                              ('l'  - static_cast<char>( 3)),
                              ('o'  - static_cast<char>( 4)),
                              (','  - static_cast<char>( 5)),
                              (' '  - static_cast<char>( 6)),
                              ('W'  - static_cast<char>( 7)),
                              ('o'  - static_cast<char>( 8)),
                              ('r'  - static_cast<char>( 9)),
                              ('l'  - static_cast<char>(10)),
                              ('d'  - static_cast<char>(11)),
                              ('!'  - static_cast<char>(12)),
                              ('\0' - static_cast<char>(13)) };
const char refdata2_arr[] = { ('H'  + static_cast<char>( 0)),
                              ('e'  + static_cast<char>( 1)),
                              ('l'  + static_cast<char>( 2)),
                              ('l'  + static_cast<char>( 3)),
                              ('o'  + static_cast<char>( 4)),
                              (','  + static_cast<char>( 5)),
                              (' '  + static_cast<char>( 6)),
                              ('W'  + static_cast<char>( 7)),
                              ('o'  + static_cast<char>( 8)),
                              ('r'  + static_cast<char>( 9)),
                              ('l'  + static_cast<char>(10)),
                              ('d'  + static_cast<char>(11)),
                              ('!'  + static_cast<char>(12)),
                              ('\0' + static_cast<char>(13)) };
CREATE_BUFFER(initdata, initdata_arr);
CREATE_BUFFER(refdata1, "Hello, World!");
CREATE_BUFFER(refdata2, refdata2_arr);



hpx::opencl::program
remotely_create_program ( hpx::opencl::device device )
{
    return device.create_program_with_source(program_src);
}

hpx::opencl::kernel
remotely_create_kernel ( hpx::opencl::program program )
{
    return program.create_kernel("hello_world");
}

hpx::opencl::buffer
remotely_create_buffer ( hpx::opencl::device device )
{
    return device.create_buffer(CL_MEM_READ_WRITE, DATASIZE);
}

HPX_PLAIN_ACTION(remotely_create_program, create_program_action);
HPX_PLAIN_ACTION(remotely_create_kernel, create_kernel_action);
HPX_PLAIN_ACTION(remotely_create_buffer, create_buffer_action);


static void remote_test( hpx::opencl::device cldevice )
{

    // get location id
    auto locality = hpx::get_colocation_id(hpx::launch::sync, cldevice.get_id());

    // remotely create a program
    hpx::opencl::program program =
        hpx::async<create_program_action>(locality, cldevice).get();

    // build program
    program.build();

    // remotely create a kernel
    hpx::opencl::kernel kernel =
        hpx::async<create_kernel_action>(locality, program).get();

    // remotely create buffers
    hpx::opencl::buffer buffer_src =
        hpx::async<create_buffer_action>(locality, cldevice).get();
    hpx::opencl::buffer buffer_dst =
        hpx::async<create_buffer_action>(locality, cldevice).get();

    // test if buffer initialization worked
    {
        size_t buffer_src_size = buffer_src.size().get();
        HPX_TEST_EQ(buffer_src_size, DATASIZE);
        size_t buffer_dst_size = buffer_dst.size().get();
        HPX_TEST_EQ(buffer_dst_size, DATASIZE);
    }

    // set kernel arguments
    {
        auto future1 = kernel.set_arg_async(0, buffer_src);
        kernel.set_arg(1, buffer_dst);
        future1.get();
    }

    // set work dimensions
    hpx::opencl::work_size<1> size;
    size[0].offset = 0;
    size[0].size = DATASIZE;

    // test if kernel can get executed (blocking)
    {
        // Initialize src buffer
        buffer_src.enqueue_write(0, initdata).get();

        // Execute
        kernel.enqueue(size).get();

        // Check for correct result
        auto result_future = buffer_dst.enqueue_read(0, DATASIZE);
        COMPARE_RESULT(result_future.get(), refdata1);
    }

    // test if kernel can get executed (non-blocking)
    {
        // Send result of blocking execution to src buffer
        auto fut1 = buffer_dst.enqueue_send(buffer_src, 0, 0, DATASIZE);

        // Execute
        auto fut2 = kernel.enqueue(size, fut1.dst_future);

        // Read data
        auto result_future = buffer_dst.enqueue_read(0, DATASIZE, fut2);
        COMPARE_RESULT(result_future.get(), refdata2);
    }

}

static void cl_test( hpx::opencl::device local_device,
                     hpx::opencl::device cldevice )
{

    remote_test(cldevice);
    remote_test(local_device);

}


