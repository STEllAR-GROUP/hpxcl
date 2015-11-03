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

#define DATASIZE (sizeof(INITDATA))


CREATE_BUFFER(initdata, INITDATA);
CREATE_BUFFER(targetdata, TARGETDATA);
CREATE_BUFFER(refdata01, REFDATA01);
CREATE_BUFFER(refdata02, REFDATA02);
CREATE_BUFFER(refdata03, REFDATA03);
CREATE_BUFFER(refdata04, REFDATA04);

#define test_send(props, ref) {                                                 \
        buffer.enqueue_write(0, initdata).get();                                \
        remote_buffer.enqueue_write(0, targetdata).get();                       \
        auto data_send_future = buffer.enqueue_send_rect(remote_buffer, props); \
        auto data_read_future = remote_buffer.enqueue_read( 0, DATASIZE,        \
                                                  data_send_future.dst_future );\
        COMPARE_RESULT(data_read_future.get(), ref);                            \
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
    test_send(props01, refdata01);
    test_send(props02, refdata02);
    test_send(props03, refdata03);
    test_send(props04, refdata04);

}


