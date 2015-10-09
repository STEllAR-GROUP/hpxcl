// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "cl_tests.hpp"


/*
 * This test is meant to verify the buffer rect write functionality.
 */


// 4x4x4 cube of data
#define INITDATA "0123" "4567" "8902" "4680" \
                 "1357" "9152" "6374" "8597" \
                 "9876" "5432" "1045" "3627" \
                 "1894" "1928" "3465" "8709"

#define TARGETDATA "1111" "1111" "1111" "1111" \
                   "1111" "1111" "1111" "1111" \
                   "1111" "1111" "1111" "1111" \
                   "1111" "1111" "1111" "1111"

#define REFDATA01 "0123" "4567" "8902" "4680" \
                  "1357" "9152" "6374" "8597" \
                  "9876" "5432" "1045" "3627" \
                  "1894" "1928" "3465" "8709"
hpx::opencl::rect_props props01(0,0,0, 0,0,0, 4,4,4, 4,16, 4,16);

#define REFDATA02 "1111" "1111" "1111" "1111" \
                  "1111" "1111" "1111" "1111" \
                  "1111" "4567" "1111" "1111" \
                  "1111" "1111" "1111" "1111"
hpx::opencl::rect_props props02(0,1,0, 0,1,2, 4,1,1, 4,16, 4,16);

#define REFDATA03 "1111" "1111" "1111" "1111" \
                  "1112" "1114" "1116" "1118" \
                  "1111" "1111" "1111" "1111" \
                  "1111" "1111" "1111" "1111"
hpx::opencl::rect_props props03(2,0,0, 3,0,1, 1,4,1, 2,0, 4,16);

#define REFDATA04 "1111" "1111" "1111" "1111" \
                  "1111" "1111" "1111" "1111" \
                  "1111" "1111" "1101" "1123" \
                  "1111" "1111" "1145" "1167"
hpx::opencl::rect_props props04(0,0,0, 2,2,2, 2,2,2, 2,4, 4,16);

static const uint16_t refdata05[] = {0x4141, 0x3233, 0x4141, 0x3832,
                                     0x4141, 0x4141, 0x4141, 0x4141};
hpx::opencl::rect_props props05(1,1,2, 0,1,0, 1,1,2, 2,8, 1,2);


#define DATASIZE (sizeof(INITDATA))


CREATE_BUFFER(initdata, INITDATA);
CREATE_BUFFER(targetdata, TARGETDATA);
CREATE_BUFFER(refdata01, REFDATA01);
CREATE_BUFFER(refdata02, REFDATA02);
CREATE_BUFFER(refdata03, REFDATA03);
CREATE_BUFFER(refdata04, REFDATA04);

static const uint16_t INT16TARGETDATA[] = {0x4141, 0x4141, 0x4141, 0x4141,
                                           0x4141, 0x4141, 0x4141, 0x4141};

#define test_read(props, ref) {                                                 \
        buffer.enqueue_write(0, initdata).get();                                \
        buffer_type target_buffer( TARGETDATA, sizeof(TARGETDATA),              \
                                   buffer_type::init_mode::copy );              \
        auto data_read_future = buffer.enqueue_read_rect(props, target_buffer); \
        COMPARE_RESULT(data_read_future.get(), ref);                            \
}
    

#define test_read_int16(props, ref) {                                           \
        buffer.enqueue_write(0, initdata).get();                                \
        int16buffer_type target_buffer( INT16TARGETDATA, 8,                     \
                                        int16buffer_type::init_mode::copy );    \
        auto data_read_future = buffer.enqueue_read_rect(props, target_buffer); \
        auto result_buffer = data_read_future.get();                            \
        HPX_TEST_EQ(result_buffer.size(), target_buffer.size());                \
        for(std::size_t i = 0; i < result_buffer.size(); i++){                  \
            HPX_TEST_EQ(result_buffer[i], target_buffer[i]);                    \
            HPX_TEST_EQ(result_buffer[i], ref[i]);                              \
        }                                                                       \
}


static void cl_test( hpx::opencl::device local_device,
                     hpx::opencl::device remote_device )
{

    hpx::opencl::buffer buffer =
        remote_device.create_buffer(CL_MEM_READ_WRITE, DATASIZE);

    hpx::opencl::buffer remote_buffer =
        local_device.create_buffer(CL_MEM_READ_WRITE, DATASIZE);

    // test if buffer initialization worked
    size_t buffer_size = buffer.size().get();
    HPX_TEST_EQ(buffer_size, DATASIZE);

    // test if buffer can be read from to
    test_read(props01, refdata01);
    test_read(props02, refdata02);
    test_read(props03, refdata03);
    test_read(props04, refdata04);
    test_read_int16(props05, refdata05);

}


