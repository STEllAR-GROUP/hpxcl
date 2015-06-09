// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cl_tests.hpp"


/*
 * This test is meant to verify the buffer read and buffer write functionality.
 */


typedef hpx::serialization::serialize_buffer<char> buffer_type;

#define DATASIZE (sizeof("Hello World!"))

#define CREATE_BUFFER(name, data)                                               \
    static const buffer_type name(data, sizeof(data),                           \
                                  buffer_type::init_mode::reference)

#define COMPARE_RESULT( result, correct_result )                                \
    HPX_TEST( strcmp( result.get().data(), correct_result.data() ) == 0 )

CREATE_BUFFER(initdata, "Hello World!");
CREATE_BUFFER(refdata1, "Help, World!");
CREATE_BUFFER(refdata2, "World");
CREATE_BUFFER(refdata3, "Hello Wolp,!");

static const buffer_type modifydata("p,", 2,
                                  buffer_type::init_mode::reference);


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
        auto data_write_future = buffer.enqueue_write(0, initdata);
        data_write_future.wait();
    }

    // test when_all
    {
        auto future1 = buffer.enqueue_write(0, initdata);
        auto future2 = buffer2.enqueue_write(0, initdata);
        
        std::vector<hpx::future<void> > futures;
        futures.push_back(std::move(future1));
        futures.push_back(std::move(future2));

        hpx::when_all(futures).get();
    }

    // test local continuation
    {
        auto data_write_future = buffer.enqueue_write(0, initdata);
        auto future2 = data_write_future.then(
            [](hpx::future<void> fut){
                return true;   
            }
        );
        HPX_TEST(future2.get());
    }

    // test read
    {
        auto data_read_future = buffer.enqueue_read(0, DATASIZE);

        COMPARE_RESULT(data_read_future, initdata);
     }

    // test remote continuation
    {
        auto data_write_future = buffer.enqueue_write(3, modifydata);
        auto data_read_future = buffer.enqueue_read(0, DATASIZE,
                                                    data_write_future);

        COMPARE_RESULT(data_read_future, refdata1);
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


