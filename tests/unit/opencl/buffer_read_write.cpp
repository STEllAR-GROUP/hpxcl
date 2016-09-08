// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cl_tests.hpp"


/*
 * This test is meant to verify the buffer read and buffer write functionality.
 */



#define DATASIZE (sizeof("Hello World!"))


CREATE_BUFFER(initdata, "Hello World!");
CREATE_BUFFER(refdata1, "Help, World!");
CREATE_BUFFER(refdata2, "World");
CREATE_BUFFER(refdata3, "Hello Wolp,!");
CREATE_BUFFER(refdata4, "HDEFGjihgld!");
CREATE_BUFFER(refdata5, "Helello rld!");


static const buffer_type modifydata("p,", 2, buffer_type::init_mode::reference);

static const uint32_t intarr[] = {0x47464544, 0x6768696a};
static const intbuffer_type modifydata2(intarr, 2,
                                        intbuffer_type::init_mode::reference);


static void cl_test( hpx::opencl::device local_device,
                     hpx::opencl::device remote_device )
{

    hpx::opencl::buffer buffer =
        remote_device.create_buffer(CL_MEM_READ_WRITE, DATASIZE);
    hpx::opencl::buffer buffer2 =
        remote_device.create_buffer(CL_MEM_READ_WRITE, DATASIZE);

    hpx::opencl::buffer remote_buffer =
        local_device.create_buffer(CL_MEM_READ_WRITE, DATASIZE);

    // test if buffer initialization worked
    size_t buffer_size = buffer.size().get();
    HPX_TEST_EQ(buffer_size, DATASIZE);

    // test if buffer can be written to
    {
        auto data_write_future = buffer.enqueue_write(0, initdata);
        data_write_future.get();
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

        COMPARE_RESULT(data_read_future.get(), initdata);
    }

    // test remote continuation
    {
        auto data_write_future = buffer.enqueue_write(3, modifydata);
        auto data_read_future = buffer.enqueue_read(0, DATASIZE,
                                                    data_write_future);

        COMPARE_RESULT(data_read_future.get(), refdata1);
    }

    // test read continuation and non-char buffer writes and offsets
    {
        auto data_read_future1 = buffer.enqueue_read(3, 2);
        auto data_write_future = buffer.enqueue_write(1, modifydata2,
                                                      data_read_future1);
        auto data_read_future2 = buffer.enqueue_read(0, DATASIZE,
                                                     data_write_future);

        COMPARE_RESULT(data_read_future1.get(), modifydata);
        COMPARE_RESULT(data_read_future2.get(), refdata4);
    }

    // test read to buffer
    {
        intbuffer_type readbuffer( 2 );

        auto data_read_future = buffer.enqueue_read(1, readbuffer);

        auto result_buffer = data_read_future.get();

        HPX_TEST(readbuffer.data() == result_buffer.data());
        HPX_TEST(readbuffer.size() == result_buffer.size());

        COMPARE_RESULT_INT(result_buffer, modifydata2);
    }

    // test buffer-buffer copy local
    {
        // write to both src and dst buffer
        auto data_write_future = buffer.enqueue_write(0, initdata);
        auto data_write_future2 = buffer2.enqueue_write(0, initdata);

        // send src to dst buffer with offset 
        auto futures = buffer.enqueue_send( buffer2, 1, 3, 5,
                                            data_write_future,
                                            data_write_future2);

        // read the src and dst buffer
        auto src_data = buffer.enqueue_read(0, DATASIZE, futures.src_future);
        auto dst_data = buffer2.enqueue_read(0, DATASIZE, futures.dst_future);

        COMPARE_RESULT(src_data.get(), initdata);
        COMPARE_RESULT(dst_data.get(), refdata5);
    }

    // test buffer-buffer copy remote
    {
        // write to both src and dst buffer
        auto data_write_future = buffer.enqueue_write(0, initdata);
        auto data_write_future2 = remote_buffer.enqueue_write(0, initdata);

        // send src to dst buffer with offset 
        auto futures = buffer.enqueue_send( remote_buffer, 1, 3, 5,
                                            data_write_future,
                                            data_write_future2);

        // read the src and dst buffer
        auto src_data = buffer.enqueue_read(0, DATASIZE, futures.src_future);
        auto dst_data = remote_buffer.enqueue_read(0, DATASIZE, futures.dst_future);

        COMPARE_RESULT(src_data.get(), initdata);
        COMPARE_RESULT(dst_data.get(), refdata5);
    }


}


