// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "util/cl_tests.hpp"

#include "util/testresults.hpp"

#include <hpx/util/high_resolution_timer.hpp>

#include <cstdlib>

typedef hpx::serialization::serialize_buffer<char> buffer_type;

// global variables
static buffer_type test_data;

buffer_type loopback(buffer_type buf) { return buf; }

HPX_PLAIN_ACTION(loopback, loopback_action);

static void ensure_valid(buffer_type result) {
  if (result.size() != test_data.size()) {
    die("result size is wrong!");
  }

  for (std::size_t i = 0; i < result.size(); i++) {
    if (test_data[i] != result[i]) die("result is wrong!");
  }
}

static void send_test(hpx::opencl::device device1, hpx::opencl::device device2,
                      bool sync) {
  hpx::opencl::buffer buffer1 =
      device1.create_buffer(CL_MEM_READ_WRITE, test_data.size());
  hpx::opencl::buffer buffer2 =
      device2.create_buffer(CL_MEM_READ_WRITE, test_data.size());

  std::string name = "send_";

  if (hpx::get_colocation_id(hpx::launch::sync, device1.get_id()) ==
      hpx::find_here())
    name += "local_";
  else
    name += "remote_";

  if (hpx::get_colocation_id(hpx::launch::sync, device2.get_id()) ==
      hpx::find_here())
    name += "local";
  else
    name += "remote";

  if (sync) name += "_sync";

  std::map<std::string, std::string> atts;
  //    atts["size"] = std::to_string(test_data.size());
  atts["iterations"] = std::to_string(num_iterations);
  results.start_test(name, "ms", atts);

  while (results.needs_more_testing()) {
    // initialize the buffer
    hpx::future<void> fut = buffer1.enqueue_write(0, test_data);

    fut.wait();

    // RUN!
    hpx::util::high_resolution_timer walltime;
    for (std::size_t it = 0; it < num_iterations; it++) {
      // Copy from buffer1 to buffer2
      auto send_result =
          buffer1.enqueue_send(buffer2, 0, 0, test_data.size(), fut);

      fut = std::move(send_result.dst_future);

      if (sync) fut.wait();
    }

    // wait for last send to finish
    fut.get();

    // Measure elapsed time
    const double duration = walltime.elapsed();

    // Check if data is still valid
    ensure_valid(buffer2.enqueue_read(0, test_data.size()).get());

    // Calculate overhead
    const double overhead = duration * 1000.0 / num_iterations;

    results.add(overhead);
  }
}

static void wait_test(hpx::opencl::device device) {
  hpx::opencl::buffer buffer =
      device.create_buffer(CL_MEM_READ_WRITE, test_data.size());

  std::string name = "wait_";

  if (hpx::get_colocation_id(hpx::launch::sync, device.get_id()) ==
      hpx::find_here())
    name += "local";
  else
    name += "remote";

  std::map<std::string, std::string> atts;
  //    atts["size"] = std::to_string(test_data.size());
  atts["iterations"] = std::to_string(num_iterations);
  results.start_test(name, "ms", atts);

  while (results.needs_more_testing()) {
    // initialize the buffer
    buffer_type write_buf1(test_data.size());
    buffer_type write_buf2(test_data.size());
    std::copy(test_data.data(), test_data.data() + test_data.size(),
              write_buf1.data());
    std::copy(test_data.data(), test_data.data() + test_data.size(),
              write_buf2.data());

    double duration = 0.0;

    // RUN!
    for (std::size_t it = 0; it < num_iterations; it++) {
      // Copy to device
      auto fut1 = buffer.enqueue_write(0, write_buf1);

      // Copy to device again, with dependency to fut1
      auto fut2 = buffer.enqueue_write(0, write_buf2, fut1);

      // wait for fut2
      fut2.get();

      // fut1 is definitely ready now, but unchecked.
      // now measure how long it takes to check fut1
      hpx::util::high_resolution_timer walltime;
      fut1.get();
      duration += walltime.elapsed();
    }

    // Calculate overhead
    const double overhead = duration * 1000.0 / num_iterations;

    results.add(overhead);
  }
}

static void write_test(hpx::opencl::device device, bool sync) {
  hpx::opencl::buffer buffer =
      device.create_buffer(CL_MEM_READ_WRITE, test_data.size());

  std::string name = "write_";

  if (hpx::get_colocation_id(hpx::launch::sync, device.get_id()) ==
      hpx::find_here())
    name += "local";
  else
    name += "remote";

  if (sync) name += "_sync";

  std::map<std::string, std::string> atts;
  //    atts["size"] = std::to_string(test_data.size());
  atts["iterations"] = std::to_string(num_iterations);
  results.start_test(name, "ms", atts);

  while (results.needs_more_testing()) {
    // initialize the buffer
    buffer_type write_buf(test_data.size());
    std::copy(test_data.data(), test_data.data() + test_data.size(),
              write_buf.data());

    hpx::future<void> fut;
    bool is_first_iteration = true;

    // RUN!
    hpx::util::high_resolution_timer walltime;
    for (std::size_t it = 0; it < num_iterations; it++) {
      if (is_first_iteration) {
        // Copy to device
        fut = buffer.enqueue_write(0, write_buf);
        is_first_iteration = false;
      } else {
        // Copy to device
        fut = buffer.enqueue_write(0, write_buf, fut);
      }

      // wait
      if (sync) fut.wait();
    }

    // wait for finish
    fut.get();

    // Measure elapsed time
    const double duration = walltime.elapsed();

    // Check if data is still valid
    ensure_valid(buffer.enqueue_read(0, test_data.size()).get());

    // Calculate overhead
    const double overhead = duration * 1000.0 / num_iterations;

    results.add(overhead);
  }
}

static void read_test(hpx::opencl::device device, bool sync) {
  hpx::opencl::buffer buffer =
      device.create_buffer(CL_MEM_READ_WRITE, test_data.size());

  std::string name = "read_";

  if (hpx::get_colocation_id(hpx::launch::sync, device.get_id()) ==
      hpx::find_here())
    name += "local";
  else
    name += "remote";

  if (sync) name += "_sync";

  std::map<std::string, std::string> atts;
  //    atts["size"] = std::to_string(test_data.size());
  atts["iterations"] = std::to_string(num_iterations);
  results.start_test(name, "ms", atts);

  while (results.needs_more_testing()) {
    // initialize the buffer
    buffer_type write_buf(test_data.size());
    std::copy(test_data.data(), test_data.data() + test_data.size(),
              write_buf.data());

    buffer.enqueue_write(0, write_buf).get();

    hpx::future<buffer_type> fut;
    bool is_first_iteration = true;

    // RUN!
    hpx::util::high_resolution_timer walltime;
    for (std::size_t it = 0; it < num_iterations; it++) {
      if (is_first_iteration) {
        // Copy from device
        fut = buffer.enqueue_read(0, write_buf);
        is_first_iteration = false;
      } else {
        // Copy from device
        fut = buffer.enqueue_read(0, write_buf, fut);
      }

      // wait
      if (sync) fut.wait();
    }

    // wait for finish
    write_buf = fut.get();

    // Measure elapsed time
    const double duration = walltime.elapsed();

    // Check if data is still valid
    ensure_valid(write_buf);

    // Calculate throughput
    const double overhead = duration * 1000.0 / num_iterations;

    results.add(overhead);
  }
}

static void cl_test(hpx::opencl::device local_device,
                    hpx::opencl::device remote_device, bool distributed) {
  testdata_size = 1;

  if (num_iterations == 0) num_iterations = 50;

  // Get localities
  hpx::naming::id_type remote_location =
      hpx::get_colocation_id(hpx::launch::sync, remote_device.get_id());
  hpx::naming::id_type local_location =
      hpx::get_colocation_id(hpx::launch::sync, local_device.get_id());
  if (local_location != hpx::find_here())
    die("Internal ERROR! local_location is not here.");

  // Generate random vector
  std::cerr << "Generating test data ..." << std::endl;
  test_data = buffer_type(testdata_size);
  std::cerr << "Test data generated." << std::endl;
  for (std::size_t i = 0; i < testdata_size; i++) {
    test_data[i] = static_cast<char>(rand());
  }

  // Run write test
  write_test(local_device, false);
  write_test(local_device, true);

  // Run read test
  read_test(local_device, false);
  read_test(local_device, true);

  // Run read test
  send_test(local_device, local_device, false);
  send_test(local_device, local_device, true);

  // Run wait test
  wait_test(local_device);

  if (distributed) {
    // Run write test
    write_test(remote_device, false);
    write_test(remote_device, true);

    // Run read test
    read_test(remote_device, false);
    read_test(remote_device, true);

    // Run read test
    send_test(local_device, remote_device, false);
    send_test(local_device, remote_device, true);
    send_test(remote_device, local_device, false);
    send_test(remote_device, local_device, true);
    send_test(remote_device, remote_device, false);
    send_test(remote_device, remote_device, true);

    // Run wait test
    wait_test(remote_device);
  }
}
