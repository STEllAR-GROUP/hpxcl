// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cl_tests.hpp"


/*
 * This test is meant to verify the kernel creation and execution functionality.
 */

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
                              (' '  - static_cast<char>( 5)), 
                              ('W'  - static_cast<char>( 6)), 
                              ('o'  - static_cast<char>( 7)), 
                              ('r'  - static_cast<char>( 8)), 
                              ('l'  - static_cast<char>( 9)), 
                              ('d'  - static_cast<char>(10)), 
                              ('!'  - static_cast<char>(11)), 
                              ('\0' - static_cast<char>(12)) };
CREATE_BUFFER(initdata, initdata_arr);
CREATE_BUFFER(refdata1, "Hello, World!");

static void create_and_run_kernel(hpx::opencl::program program){

/*

    // test if kernel can be created
    hpx::opencl::kernel kernel = program.create_kernel("hello_world");

    // create source and destination buffers
    hpx::opencl::buffer buffer_src =
        cldevice.create_buffer(CL_MEM_READ_WRITE, DATASIZE);
    hpx::opencl::buffer buffer_dst =
        cldevice.create_buffer(CL_MEM_READ_WRITE, DATASIZE);

    // test if buffer initialization worked
    {
        size_t buffer_src_size = buffer_src.size().get();
        HPX_TEST_EQ(buffer_src_size, DATASIZE);
        size_t buffer_dst_size = buffer_dst.size().get();
        HPX_TEST_EQ(buffer_dst_size, DATASIZE);
    }
*/
    
    // set kernel arguments
    // TODO
    
    // test if kernel can get executed
    // TODO
    
    // test if computation result is correct
    // TODO
    

}

static void cl_test( hpx::opencl::device local_device,
                     hpx::opencl::device cldevice )
{

    {
        // test if program can be created from source
        hpx::opencl::program program =
            cldevice.create_program_with_source(program_src);

        // test if program can be compiled
        //program.build().wait();

    }

    // same test with build arguments
    // TODO

    // test with create_from_binary
    // TODO

}


