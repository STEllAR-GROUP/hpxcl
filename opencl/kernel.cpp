// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Header File
#include "kernel.hpp"

// Internal Dependencies
#include "server/kernel.hpp"
#include "buffer.hpp"

using hpx::opencl::kernel;

void kernel::ensure_device_id() const {
  if (!device_gid) {
    typedef hpx::opencl::server::kernel::get_parent_device_id_action
        action_type;
    HPX_ASSERT(this->get_id());
    device_gid = async<action_type>(this->get_id()).get();
  }
}

void kernel::set_arg(cl_uint arg_index, const hpx::opencl::buffer &arg) const {
  set_arg_async(arg_index, arg).get();
}

hpx::lcos::future<void> kernel::set_arg_async(
    cl_uint arg_index, const hpx::opencl::buffer &arg) const {
  HPX_ASSERT(this->get_id());

  typedef hpx::opencl::server::kernel::set_arg_action func;

  return hpx::async<func>(this->get_id(), arg_index, arg.get_id());
}

hpx::future<void> kernel::enqueue_impl(
    std::vector<std::size_t> &&size_vec,
    hpx::opencl::util::resolved_events &&deps) const {
  // create local event
  using hpx::opencl::lcos::event;
  event<void> ev(device_gid);

  // send command to server class
  typedef hpx::opencl::server::kernel::enqueue_action func;
  hpx::apply<func>(this->get_id(), std::move(ev.get_event_id()), size_vec,
                   std::move(deps.event_ids));

  // return future connected to event
  return ev.get_future();
}
