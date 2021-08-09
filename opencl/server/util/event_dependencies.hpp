// Copyright (c)    2015 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_UTIL_EVENT_DEPENDENCIES_HPP_
#define HPX_OPENCL_SERVER_UTIL_EVENT_DEPENDENCIES_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "../../cl_headers.hpp"
#include "../../fwd_declarations.hpp"

////////////////////////////////////////////////////////////////
namespace hpx {
namespace opencl {
namespace server {
namespace util {

////////////////////////////////////////////////////////
// This class is used to convert event ids to cl_events
//
class HPX_OPENCL_EXPORT event_dependencies {
 public:
  // Constructor
  event_dependencies(const std::vector<hpx::naming::id_type>& event_ids,
                     hpx::opencl::server::device* parent_device);
  ~event_dependencies();

  //////////////////////////////////////////////////
  /// Local public functions
  ///

  // Returns a pointer to a list of cl_events. Ensure that deallocation
  // of this class only happens when this pointer is not needed any more!
  //
  // Returns NULL if size() == 0
  cl_event* get_cl_events();

  // Returns the number of events in this list
  std::size_t size();

 private:
  std::vector<cl_event> events;
};
}  // namespace util
}  // namespace server
}  // namespace opencl
}  // namespace hpx

#endif
