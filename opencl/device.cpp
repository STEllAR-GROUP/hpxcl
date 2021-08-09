// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The class header file
#include "device.hpp"

// The server class
#include "server/device.hpp"

// Dependencies
#include "buffer.hpp"
#include "program.hpp"
#include "util/generic_buffer.hpp"

using hpx::opencl::device;

hpx::opencl::util::generic_buffer device::get_device_info_raw(
    cl_device_info info_type) const {
  HPX_ASSERT(this->get_id());

  typedef hpx::opencl::server::device::get_device_info_action func;

  return hpx::opencl::util::generic_buffer(
      hpx::async<func>(this->get_id(), info_type));
}

hpx::opencl::util::generic_buffer device::get_platform_info_raw(
    cl_platform_info info_type) const {
  HPX_ASSERT(this->get_id());

  typedef hpx::opencl::server::device::get_platform_info_action func;

  return hpx::opencl::util::generic_buffer(
      hpx::async<func>(this->get_id(), info_type));
}

hpx::opencl::buffer device::create_buffer(cl_mem_flags flags,
                                          std::size_t size) const {
  HPX_ASSERT(this->get_id());

  typedef hpx::opencl::server::device::create_buffer_action func;

  hpx::future<hpx::id_type> buffer_server =
      hpx::async<func>(this->get_id(), flags, size);

  return buffer(std::move(buffer_server), this->get_id());
}

hpx::opencl::program device::create_program_with_source(
    hpx::serialization::serialize_buffer<char> src) const {
  HPX_ASSERT(this->get_id());

  typedef hpx::opencl::server::device::create_program_with_source_action func;

  hpx::future<hpx::id_type> program_server =
      hpx::async<func>(this->get_id(), src);

  return program(std::move(program_server), this->get_id());
}

hpx::opencl::program device::create_program_with_binary(
    hpx::serialization::serialize_buffer<char> binary) const {
  HPX_ASSERT(this->get_id());

  typedef hpx::opencl::server::device::create_program_with_binary_action func;

  hpx::future<hpx::id_type> program_server =
      hpx::async<func>(this->get_id(), binary);

  return program(std::move(program_server), this->get_id());
}
