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

static const char modifydata[] = "p,";
static const char refdata1[] = "Help, World!";

static const char refdata2[] = "World";


static void cl_test()
{

    hpx::opencl::buffer buffer = cldevice->create_buffer(CL_MEM_READ_WRITE,
                                                        DATASIZE,
                                                        initdata);

    // test if buffer initialization worked
    size_t buffer_size = buffer.size().get();
    HPX_TEST_EQ(buffer_size, DATASIZE);

    // read and compare
    TEST_CL_BUFFER(buffer, initdata);

    // write to buffer
    buffer.enqueue_write(3, 2, modifydata).get().await();

    // read and compare
    TEST_CL_BUFFER(buffer, refdata1);
    
    // test offsetted read
    boost::shared_ptr<std::vector<char>> out = 
                               buffer.enqueue_read(6, 5).get().get_data().get();
    HPX_TEST_EQ(std::string(refdata2), std::string(out->begin(), out->end()));

}


