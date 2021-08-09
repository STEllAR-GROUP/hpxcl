// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Header File
#include "program.hpp"

// Internal Dependencies
#include "server/program.hpp"

#include "kernel.hpp"

using hpx::opencl::program;

void program::ensure_device_id() const {
  if (!device_gid) {
    typedef hpx::opencl::server::program::get_parent_device_id_action
        action_type;
    HPX_ASSERT(this->get_id());
    device_gid = async<action_type>(this->get_id()).get();
  }
}

void program::build() const { build_async("").get(); }

void program::build(std::string build_options) const {
  build_async(std::move(build_options)).get();
}

hpx::lcos::future<void> program::build_async() const { return build_async(""); }

hpx::lcos::future<void> program::build_async(std::string build_options) const {
  HPX_ASSERT(this->get_id());

  typedef hpx::opencl::server::program::build_action func;

  return async<func>(this->get_id(), build_options);
}

hpx::lcos::future<hpx::serialization::serialize_buffer<char> >
program::get_binary() const {
  HPX_ASSERT(this->get_id());

  typedef hpx::opencl::server::program::get_binary_action func;

  return async<func>(this->get_id());
}

hpx::opencl::kernel program::create_kernel(std::string kernel_name) const {
  HPX_ASSERT(this->get_id());

  typedef hpx::opencl::server::program::create_kernel_action func;

  hpx::future<hpx::id_type> kernel_server =
      hpx::async<func>(this->get_id(), kernel_name);

  ensure_device_id();
  return kernel(std::move(kernel_server), device_gid);
}
