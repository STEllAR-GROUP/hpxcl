// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_DEVICE_HPP_
#define HPX_OPENCL_SERVER_DEVICE_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "../export_definitions.hpp"
#include "../cl_headers.hpp"

#include "../fwd_declarations.hpp"

#include "util/event_map.hpp"
#include "util/data_map.hpp"

// REGISTER_ACTION_DECLARATION templates
#include "util/server_definitions.hpp"

////////////////////////////////////////////////////////////////
namespace hpx {
namespace opencl {
namespace server {

// /////////////////////////////////////////////////////
// This class represents an OpenCL accelerator device.
//
class HPX_OPENCL_EXPORT device
    : public hpx::components::managed_component_base<device> {
  typedef hpx::lcos::local::spinlock lock_type;
  typedef hpx::serialization::serialize_buffer<char> buffer_type;

 public:
  // Constructor
  device();
  ~device();

  //////////////////////////////////////////////////
  /// Local public functions
  ///
  void init(cl_device_id device_id, bool enable_profiling = false);

  cl_context get_context();
  cl_device_id get_device_id();

  //////////////////////////////////////////////////
  /// Exposed functionality of this component
  ///

  // returns device specific information
  hpx::serialization::serialize_buffer<char> get_device_info(
      cl_device_info info_type);

  // returns platform specific information
  hpx::serialization::serialize_buffer<char> get_platform_info(
      cl_platform_info info_type);

  // creates a new buffer
  hpx::id_type create_buffer(cl_mem_flags flags, std::size_t size);

  // creates a new program from source
  hpx::id_type create_program_with_source(
      hpx::serialization::serialize_buffer<char>);

  // creates a new program from binary
  hpx::id_type create_program_with_binary(
      hpx::serialization::serialize_buffer<char>);

  /////////////////////////////////////////////////
  /// Behind-the-scenes functionality of this component
  ///

  // releases an event registered to a GID
  void release_event(hpx::naming::gid_type);

  // activates a deferred event
  void activate_deferred_event(hpx::naming::id_type);

  // activates a deferred event<serialize_buffer<char> >
  void activate_deferred_event_with_data(hpx::naming::id_type);

  HPX_DEFINE_COMPONENT_ACTION(device, get_device_info);
  HPX_DEFINE_COMPONENT_ACTION(device, get_platform_info);
  HPX_DEFINE_COMPONENT_ACTION(device, create_buffer);
  HPX_DEFINE_COMPONENT_ACTION(device, create_program_with_source);
  HPX_DEFINE_COMPONENT_ACTION(device, create_program_with_binary);
  HPX_DEFINE_COMPONENT_ACTION(device, release_event);
  HPX_DEFINE_COMPONENT_ACTION(device, activate_deferred_event);

 public:
  /////////////////////////////////////////////////
  // Public Member Functions

  // registers an event-GID pair
  void register_event(const hpx::naming::id_type& gid, cl_event);

  // retrieves an event from a GID
  cl_event retrieve_event(const hpx::naming::id_type& gid);

  // command queue retrievals
  cl_command_queue get_read_command_queue();
  cl_command_queue get_write_command_queue();
  cl_command_queue get_kernel_command_queue();

  // event data handling. needed to keep clEnqueue* data alive
  // (like clEnqueueWriteBuffer)
  template <typename T>
  void put_event_data(cl_event event,
                      hpx::serialization::serialize_buffer<T> data) {
    event_data_map.add(event, data);
  }

  // Waits for an opencl event.
  // Necessary to offload wait from hpx to os thread.
  void wait_for_cl_event(cl_event);

 private:
  ///////////////////////////////////////////////
  // Private Member Functions
  //

  // Error Callback
  static void CL_CALLBACK error_callback(const char*, const void*, std::size_t,
                                         void*);

  // cl_event Deletion Callback
  static void delete_event(cl_event);

  // Releases the data that was being kept alive
  void delete_event_data(cl_event);

 private:
  ///////////////////////////////////////////////
  // Private Member Variables
  //
  cl_device_id device_id;
  cl_platform_id platform_id;
  cl_context context;
  cl_command_queue command_queue;

  util::event_map event_map;
  util::data_map event_data_map;
};
}  // namespace server
}  // namespace opencl
}  // namespace hpx

//[opencl_management_registration_declarations
HPX_OPENCL_REGISTER_ACTION_DECLARATION(device, get_device_info);
HPX_OPENCL_REGISTER_ACTION_DECLARATION(device, get_platform_info);
HPX_OPENCL_REGISTER_ACTION_DECLARATION(device, create_buffer);
HPX_OPENCL_REGISTER_ACTION_DECLARATION(device, create_program_with_source);
HPX_OPENCL_REGISTER_ACTION_DECLARATION(device, create_program_with_binary);
HPX_OPENCL_REGISTER_ACTION_DECLARATION(device, release_event);
HPX_OPENCL_REGISTER_ACTION_DECLARATION(device, activate_deferred_event);
//]

#endif
