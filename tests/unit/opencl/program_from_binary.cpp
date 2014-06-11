// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cl_tests.hpp"


/*
 * This test is meant to verify the buffer read and buffer write functionality.
 */


static const char square_src[] = 
"                                                                          \n"
"   __kernel void square(__global int * val)                               \n"
"   {                                                                      \n"
"       size_t tid = get_global_id(0);                                     \n"
"       val[tid] = val[tid]*val[tid];                                      \n"
"   }                                                                      \n"
"                                                                          \n";

static const int initdata[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
#define DATASIZE ((size_t)10)

static const char refdata1[] = {0, 1, 2, 9, 16, 25, 36, 49, 8, 9};

static void cl_test(hpx::opencl::device cldevice)
{

    hpx::opencl::buffer buffer = cldevice.create_buffer(CL_MEM_READ_WRITE,
                                                              DATASIZE,
                                                              initdata);

    // test if buffer initialization worked
    size_t buffer_size = buffer.size().get();
    HPX_TEST_EQ(buffer_size, DATASIZE);

    // read and compare
    TEST_CL_BUFFER(buffer, initdata);

    // create variable to hold binary
    std::vector<char> binary;

    // Create new scope for compilation
    {
        
        // create program
        hpx::opencl::program prog = cldevice.create_program_with_source(
                                                                    square_src);

        // build program
        prog.build();

        // retrieve binary
        binary = prog.get_binary().get();
 
    }
        
    // print binary size
    hpx::cout << "Binary size: " << binary.size() << hpx::endl;

    // create program from binary
    hpx::opencl::program prog
                = cldevice.create_program_with_binary(binary.size(),
                                                      binary.data());

    // build program
    prog.build();

    // create kernel
    hpx::opencl::kernel square_kernel = prog.create_kernel("square");

    // set kernel arg
    square_kernel.set_arg(0, buffer);

    // create work_size
    hpx::opencl::work_size<1> dim;
    dim[0].offset = 3;
    dim[0].size = 5;

    // run kernel
    square_kernel.enqueue(dim).get().await();

    // test if kernel executed successfully
    TEST_CL_BUFFER(buffer, refdata1);

}


