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
                                                              DATASIZE);
    hpx::opencl::buffer buffer2 = cldevice.create_buffer(CL_MEM_READ_WRITE,
                                                              DATASIZE);

    // test if buffer initialization worked
    size_t buffer_size = buffer.size().get();
    HPX_TEST_EQ(buffer_size, DATASIZE);

    // test if buffer can be written to
    {
        auto data_write_future = buffer.enqueue_write(0, DATASIZE, initdata);
        data_write_future.wait();
    }

    // test when_all
    {
        auto future1 = buffer.enqueue_write(0, DATASIZE, initdata);
        auto future2 = buffer2.enqueue_write(0, DATASIZE, initdata);
        
        std::vector<hpx::future<void> > futures;
        futures.push_back(std::move(future1));
        futures.push_back(std::move(future2));

        hpx::when_all(futures).get();
    }

    // test local continuation
    {
        auto data_write_future = buffer.enqueue_write(0, DATASIZE, initdata);
        auto future2 = data_write_future.then(
            [](hpx::future<void> fut){
                return true;   
            }
        );
        HPX_TEST(future2.get());
    }

    // TODO local wait test, remote continuation test
/*
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

*/
}


