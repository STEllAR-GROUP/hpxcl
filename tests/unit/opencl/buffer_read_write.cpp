// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cl_tests.hpp"


/*
 * This test is meant to verify the buffer read and buffer write functionality.
 */


typedef hpx::serialization::serialize_buffer<char> buffer_type;
typedef hpx::serialization::serialize_buffer<uint32_t> intbuffer_type;

#define DATASIZE (sizeof("Hello World!"))

#define CREATE_BUFFER(name, data)                                               \
    static const buffer_type name(data, sizeof(data),                           \
                                  buffer_type::init_mode::reference)

std::string to_string(buffer_type buf){
    std::size_t length = 0; 
    while(length < buf.size())
    {
        if(buf[length] == '\0') break;
        length++;
    }
    return std::string(buf.data(), buf.data() + length);
}

#define COMPARE_RESULT( result_data, correct_result )                           \
{                                                                               \
    auto lhs = result_data;                                                     \
    auto rhs = correct_result;                                                  \
    HPX_TEST_EQ(lhs.size(), rhs.size());                                        \
    std::string correct_string = to_string(rhs);                                \
    std::string result_string = to_string(lhs);                                 \
    HPX_TEST_EQ(correct_string, result_string);                                 \
}

#define COMPARE_RESULT_INT( result_data, correct_result )                       \
{                                                                               \
    auto lhs = result_data;                                                     \
    auto rhs = correct_result;                                                  \
    HPX_TEST_EQ(lhs.size(), rhs.size());                                        \
    for(std::size_t i = 0; i < lhs.size(); i++){                                \
        std::cout << std::hex << lhs[i] << "-" << rhs[i] << std::endl;          \
        HPX_TEST_EQ(lhs[i], rhs[i]);                                            \
    }                                                                           \
}

CREATE_BUFFER(initdata, "Hello World!");
CREATE_BUFFER(refdata1, "Help, World!");
CREATE_BUFFER(refdata2, "World");
CREATE_BUFFER(refdata3, "Hello Wolp,!");
CREATE_BUFFER(refdata4, "HDEFGjihgld!");


static const buffer_type modifydata("p,", 2, buffer_type::init_mode::reference);

static const uint32_t intarr[] = {0x47464544, 0x6768696a};
static const intbuffer_type modifydata2(intarr, 2,
                                        intbuffer_type::init_mode::reference);


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
        intbuffer_type readbuffer( new uint32_t[2], 2,
                               intbuffer_type::init_mode::take );

        auto data_read_future = buffer.enqueue_read(1, readbuffer);

        auto result_buffer = data_read_future.get();

        HPX_TEST(readbuffer.data() == result_buffer.data());
        HPX_TEST(readbuffer.size() == result_buffer.size());

        COMPARE_RESULT_INT(result_buffer, modifydata2);
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


