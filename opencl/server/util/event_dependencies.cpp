// Copyright (c)    2015 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "event_dependencies.hpp"

#include "../device.hpp"

using hpx::opencl::server::util::event_dependencies;

event_dependencies::event_dependencies(
    const std::vector<hpx::naming::id_type>& event_ids,
    hpx::opencl::server::device* parent_device) {
  if (event_ids.size() != 0) {
    events.reserve(event_ids.size());

    for (const auto& id : event_ids) {
      events.push_back(parent_device->retrieve_event(id));
    }
  }
}

event_dependencies::~event_dependencies() {}

std::size_t event_dependencies::size() { return events.size(); }

cl_event* event_dependencies::get_cl_events() {
  if (events.size() == 0) return NULL;

  return events.data();
}
