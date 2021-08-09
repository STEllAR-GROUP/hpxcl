// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "cl_tests.hpp"

/*
 * This test is meant to verify the buffer rect write functionality.
 */

// 3x3x3 cube of "0"
#define INITDATA \
  "0000"         \
  "0000"         \
  "0000"         \
  "0000"         \
  "0000"         \
  "0000"         \
  "0000"         \
  "0000"         \
  "0000"         \
  "0000"         \
  "0000"         \
  "0000"         \
  "0000"         \
  "0000"         \
  "0000"         \
  "0000"

#define REFDATA01 \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0200"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"
hpx::opencl::rect_props props01(1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 4, 16);

#define REFDATA02 \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "5678"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"
hpx::opencl::rect_props props02(0, 1, 0, 0, 1, 2, 4, 1, 1, 4, 0, 4, 16);

#define REFDATA03 \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0002"          \
  "0004"          \
  "0006"          \
  "0008"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"
hpx::opencl::rect_props props03(1, 0, 0, 3, 0, 1, 1, 4, 1, 2, 0, 4, 16);

#define REFDATA04 \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0012"          \
  "0034"          \
  "0000"          \
  "0000"          \
  "0056"          \
  "0078"
hpx::opencl::rect_props props04(0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 4, 4, 16);

#define REFDATA05 \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "00DE"          \
  "0000"          \
  "0000"          \
  "0000"          \
  "00ji"          \
  "0000"          \
  "0000"
hpx::opencl::rect_props props05(0, 1, 0, 1, 1, 2, 1, 1, 2, 1, 2, 2, 8);

#define DATASIZE (sizeof(INITDATA))

CREATE_BUFFER(initdata, INITDATA);
CREATE_BUFFER(refdata01, REFDATA01);
CREATE_BUFFER(refdata02, REFDATA02);
CREATE_BUFFER(refdata03, REFDATA03);
CREATE_BUFFER(refdata04, REFDATA04);
CREATE_BUFFER(refdata05, REFDATA05);
CREATE_BUFFER(moddata, "12345678");

static const uint16_t intarr[] = {0x4746, 0x4544, 0x6768, 0x696a};
static const int16buffer_type moddata2(intarr, 4,
                                       int16buffer_type::init_mode::reference);

#define test_write(props, modifydata, ref)                                 \
  {                                                                        \
    buffer.enqueue_write(0, initdata).get();                               \
    auto data_write_future = buffer.enqueue_write_rect(props, modifydata); \
    auto data_read_future =                                                \
        buffer.enqueue_read(0, DATASIZE, data_write_future);               \
    COMPARE_RESULT(data_read_future.get(), ref);                           \
  }

static void cl_test(hpx::opencl::device local_device,
                    hpx::opencl::device remote_device) {
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
  test_write(props01, moddata, refdata01);
  test_write(props02, moddata, refdata02);
  test_write(props03, moddata, refdata03);
  test_write(props04, moddata, refdata04);
  test_write(props05, moddata2, refdata05);
}
