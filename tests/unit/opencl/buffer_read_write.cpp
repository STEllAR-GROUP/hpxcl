// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cl_tests.hpp"


/*
 * This test is meant to verify the buffer read and buffer write functionality.
 */


static const char initdata[] = "Hello World!";
#define DATASIZE ((size_t)13)

static char modifydata[] = "p,";
static const char refdata1[] = "Help, World!";

static const char refdata2[] = "World";

static const char refdata3[] = "Hello Wolp,!";

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

    // write to buffer
    hpx::lcos::future<hpx::opencl::event> write_event = 
                        buffer.enqueue_write(3, 2, modifydata);

    // change modifydata to test wether write caches internally 
//    modifydata[1] = '.';

    // wait for write to finish
    write_event.get().await();

    // read and compare
    TEST_CL_BUFFER(buffer, refdata1);
    
    // test offsetted read
    boost::shared_ptr<std::vector<char>> out = 
                               buffer.enqueue_read(6, 5).get().get_data().get();
    HPX_TEST_EQ(std::string(refdata2), std::string(out->begin(), out->end()));

    
    
    // Create second buffer
    hpx::opencl::buffer buffer2 = cldevice.create_buffer(CL_MEM_READ_WRITE,
                                                               DATASIZE,
                                                               initdata);

    // Buffer copy test
    buffer2.enqueue_copy(buffer, 2, 8, 3).get().await();

    // read and compare
    TEST_CL_BUFFER(buffer2, refdata3);


}


