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

static void create_and_run_kernel( hpx::opencl::device cldevice,
                                   hpx::opencl::program program ){

    // test if kernel can be created
    hpx::opencl::kernel kernel = program.create_kernel("hello_world");

    // test if creation of invalid kernels throws
    {
        bool caught_exception = false;
        try{
            hpx::opencl::kernel kernel = program.create_kernel("blub");
            kernel.get_gid();
        } catch (hpx::exception e){
            caught_exception = true;
        }
        HPX_ASSERT(caught_exception);
    }

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

    // standard hello-world test
    {
        // test if program can be created from source
        hpx::opencl::program program =
            cldevice.create_program_with_source(program_src);

        // test if program can be compiled
        // IMPORTANT! use get(). wait() does not throw errors.
        program.build_async().get();

        // test if program can be used for computation
        create_and_run_kernel(cldevice, program);
    }

    // same test with build arguments
    {
        // test if program can be created from source
        hpx::opencl::program program =
            cldevice.create_program_with_source(program_src);

        // test if program can be compiled
        program.build_async("-Werror").get();

        // test if program can be used for computation
        create_and_run_kernel(cldevice, program);
    }

    // test with create_from_binary
    {
        // test if program can be created from source
        hpx::opencl::program program1 =
            cldevice.create_program_with_source(program_src);

        // test if program can be compiled
        program1.build_async().get();

        // retrieve binary of program1
        auto program_binary = program1.get_binary().get();

        hpx::cout << "Binary:" << hpx::endl;
        hpx::cout << to_string(program_binary) << hpx::endl << hpx::endl;;

        // test if program can be created from binary
        hpx::opencl::program program2 =
            cldevice.create_program_with_binary(program_binary);

        // test if program can be compiled
        program2.build();

        // test if program can be used for computation
        create_and_run_kernel(cldevice, program2);
    }

    // Test compiler error detection
    {
        // create program from source. this should not throw 
        hpx::opencl::program program =
            cldevice.create_program_with_source(invalid_program_src);

        // Try to build. This should throw an error.
        bool caught_exception = false;
        try{
            program.build_async().get();
        } catch (hpx::exception e){
            hpx::cout << "Build error:" << hpx::endl;
            hpx::cout << e.what() << hpx::endl << hpx::endl;
            caught_exception = true;
        }
        HPX_TEST(caught_exception);
    }

}


