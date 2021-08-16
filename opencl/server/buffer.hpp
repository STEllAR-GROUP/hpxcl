// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_BUFFER_HPP
#define HPX_OPENCL_SERVER_BUFFER_HPP

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "../cl_headers.hpp"

#include "../fwd_declarations.hpp"

#include "../lcos/zerocopy_buffer.hpp"
#include "../lcos/event.hpp"

// REGISTER_ACTION_DECLARATION templates
#include "util/server_definitions.hpp"
#include "../util/rect_props.hpp"

namespace hpx {
namespace opencl {
namespace server {

// /////////////////////////////////////////////////////
//  This class represents an opencl buffer.

class HPX_OPENCL_EXPORT buffer
    : public hpx::components::managed_component_base<buffer> {
 public:
  // Constructor
  buffer();
  // Destructor
  ~buffer();

  ///////////////////////////////////////////////////
  /// Local functions
  ///
  void init(hpx::naming::id_type device_id, cl_mem_flags flags,
            std::size_t size);

  cl_mem get_cl_mem();

  //////////////////////////////////////////////////
  /// Exposed functionality of this component
  ///
  // Returns the size of the buffer
  std::size_t size();

  // Returns the parent device
  hpx::naming::id_type get_parent_device_id();

  // Writes to the buffer
  template <typename T>
  void enqueue_write(hpx::naming::id_type &&event_gid, std::size_t offset,
                     hpx::serialization::serialize_buffer<T> data,
                     std::vector<hpx::naming::id_type> &&dependencies);

  // Writes to the buffer
  template <typename T>
  void enqueue_write_rect(hpx::naming::id_type &&event_gid,
                          hpx::opencl::rect_props &&rect_properties,
                          hpx::serialization::serialize_buffer<T> data,
                          std::vector<hpx::naming::id_type> &&dependencies);

  // Reads from the buffer
  void enqueue_read(hpx::naming::id_type &&event_gid, std::size_t offset,
                    std::size_t size,
                    std::vector<hpx::naming::id_type> &&dependencies);

  // Reads from the buffer. Needed for direct copy to user-supplied buffer
  template <typename T>
  void enqueue_read_to_userbuffer_remote(
      hpx::naming::id_type &&event_gid, std::size_t offset, std::size_t size,
      std::uintptr_t remote_data_addr,
      std::vector<hpx::naming::id_type> &&dependencies);

  // Reads from the buffer. Needed for direct copy to user-supplied buffer
  template <typename T>
  void enqueue_read_to_userbuffer_local(
      hpx::naming::id_type &&event_gid, std::size_t offset,
      hpx::serialization::serialize_buffer<T> data,
      std::vector<hpx::naming::id_type> &&dependencies);

  // Reads from the buffer. Needed for direct copy to user-supplied buffer
  template <typename T>
  void enqueue_read_to_userbuffer_rect_remote(
      hpx::naming::id_type &&event_gid,
      hpx::opencl::rect_props &&rect_properties,
      std::uintptr_t remote_data_addr,
      std::vector<hpx::naming::id_type> &&dependencies);

  // Reads from the buffer. Needed for direct copy to user-supplied buffer
  template <typename T>
  void enqueue_read_to_userbuffer_rect_local(
      hpx::naming::id_type &&event_gid,
      hpx::opencl::rect_props &&rect_properties,
      hpx::serialization::serialize_buffer<T> data,
      std::vector<hpx::naming::id_type> &&dependencies);

  // Copies data from this buffer to a remote buffer
  void enqueue_send(hpx::naming::id_type dst, hpx::naming::id_type &&src_event,
                    hpx::naming::id_type &&dst_event, std::size_t src_offset,
                    std::size_t dst_offset, std::size_t size,
                    std::vector<hpx::naming::id_type> &&dependencies,
                    std::vector<hpx::naming::gid_type> &&dependency_devices);

  // Copies data from this buffer to a remote buffer
  void enqueue_send_rect(
      hpx::naming::id_type dst, hpx::naming::id_type &&src_event,
      hpx::naming::id_type &&dst_event, rect_props rect_properties,
      std::vector<hpx::naming::id_type> &&dependencies,
      std::vector<hpx::naming::gid_type> &&dependency_devices);

  // Different versions of enqueue_send, optimized for different
  // runtime scenarios
  void send_bruteforce(hpx::naming::id_type &&dst,
                       hpx::naming::id_type &&src_event,
                       hpx::naming::id_type &&dst_event, std::size_t src_offset,
                       std::size_t dst_offset, std::size_t size,
                       std::vector<hpx::naming::id_type> &&src_dependencies,
                       std::vector<hpx::naming::id_type> &&dst_dependencies);
  void send_direct(hpx::naming::id_type &&dst,
                   std::shared_ptr<hpx::opencl::server::buffer> &&dst_buffer,
                   hpx::naming::id_type &&src_event,
                   hpx::naming::id_type &&dst_event, std::size_t src_offset,
                   std::size_t dst_offset, std::size_t size,
                   std::vector<hpx::naming::id_type> &&src_dependencies,
                   std::vector<hpx::naming::id_type> &&dst_dependencies);
  void send_rect_bruteforce(
      hpx::naming::id_type &&dst, hpx::naming::id_type &&src_event,
      hpx::naming::id_type &&dst_event, rect_props &&rect_properties,
      std::vector<hpx::naming::id_type> &&src_dependencies,
      std::vector<hpx::naming::id_type> &&dst_dependencies);
  void send_rect_direct(
      hpx::naming::id_type &&dst,
      std::shared_ptr<hpx::opencl::server::buffer> &&dst_buffer,
      hpx::naming::id_type &&src_event, hpx::naming::id_type &&dst_event,
      rect_props &&rect_properties,
      std::vector<hpx::naming::id_type> &&src_dependencies,
      std::vector<hpx::naming::id_type> &&dst_dependencies);

  HPX_DEFINE_COMPONENT_ACTION(buffer, size);
  HPX_DEFINE_COMPONENT_ACTION(buffer, get_parent_device_id);
  HPX_DEFINE_COMPONENT_ACTION(buffer, enqueue_read);
  HPX_DEFINE_COMPONENT_ACTION(buffer, enqueue_send);
  HPX_DEFINE_COMPONENT_ACTION(buffer, enqueue_send_rect);

  // Actions with template arguments (see enqueue_write<>() above) require
  // special type definitions. The simplest way to define such an action type
  // is by deriving from the HPX facility make_action:
  template <typename T>
  struct enqueue_write_action
      : hpx::actions::make_action<
            void (buffer::*)(hpx::naming::id_type &&, std::size_t,
                             hpx::serialization::serialize_buffer<T>,
                             std::vector<hpx::naming::id_type> &&),
            &buffer::template enqueue_write<T>, enqueue_write_action<T> > {};
  template <typename T>
  struct enqueue_write_rect_action
      : hpx::actions::make_action<void (buffer::*)(
                                      hpx::naming::id_type &&,
                                      hpx::opencl::rect_props &&,
                                      hpx::serialization::serialize_buffer<T>,
                                      std::vector<hpx::naming::id_type> &&),
                                  &buffer::template enqueue_write_rect<T>,
                                  enqueue_write_rect_action<T> > {};
  template <typename T>
  struct enqueue_read_to_userbuffer_remote_action
      : hpx::actions::make_action<
            void (buffer::*)(hpx::naming::id_type &&, std::size_t, std::size_t,
                             std::uintptr_t,
                             std::vector<hpx::naming::id_type> &&),
            &buffer::template enqueue_read_to_userbuffer_remote<T>,
            enqueue_read_to_userbuffer_remote_action<T> > {};
  template <typename T>
  struct enqueue_read_to_userbuffer_local_action
      : hpx::actions::make_action<
            void (buffer::*)(hpx::naming::id_type &&, std::size_t,
                             hpx::serialization::serialize_buffer<T>,
                             std::vector<hpx::naming::id_type> &&),
            &buffer::template enqueue_read_to_userbuffer_local<T>,
            enqueue_read_to_userbuffer_local_action<T> > {};
  template <typename T>
  struct enqueue_read_to_userbuffer_rect_remote_action
      : hpx::actions::make_action<
            void (buffer::*)(hpx::naming::id_type &&,
                             hpx::opencl::rect_props &&, std::uintptr_t,
                             std::vector<hpx::naming::id_type> &&),
            &buffer::template enqueue_read_to_userbuffer_rect_remote<T>,
            enqueue_read_to_userbuffer_rect_remote_action<T> > {};
  template <typename T>
  struct enqueue_read_to_userbuffer_rect_local_action
      : hpx::actions::make_action<
            void (buffer::*)(hpx::naming::id_type &&,
                             hpx::opencl::rect_props &&,
                             hpx::serialization::serialize_buffer<T>,
                             std::vector<hpx::naming::id_type> &&),
            &buffer::template enqueue_read_to_userbuffer_rect_local<T>,
            enqueue_read_to_userbuffer_rect_local_action<T> > {};

  //////////////////////////////////////////////////
  //  Private Member Variables
  //
 private:
  std::shared_ptr<device> parent_device;
  cl_mem device_mem;
  hpx::naming::id_type parent_device_id;
};

}  // namespace server
}  // namespace opencl
}  // namespace hpx

//[opencl_management_registration_declarations
HPX_REGISTER_ACTION_DECLARATION(
    hpx::opencl::server::buffer::get_parent_device_id_action,
    hpx_opencl_buffer_get_parent_device_id_action);
HPX_OPENCL_REGISTER_ACTION_DECLARATION(buffer, size);
HPX_OPENCL_REGISTER_ACTION_DECLARATION(buffer, enqueue_read);
HPX_OPENCL_REGISTER_ACTION_DECLARATION(buffer, enqueue_send);
HPX_OPENCL_REGISTER_ACTION_DECLARATION(buffer, enqueue_send_rect);
HPX_OPENCL_TEMPLATE_ACTION_USES_MEDIUM_STACK(buffer, enqueue_write);
HPX_OPENCL_TEMPLATE_ACTION_USES_MEDIUM_STACK(buffer, enqueue_write_rect);
HPX_OPENCL_TEMPLATE_ACTION_USES_MEDIUM_STACK(buffer,
                                             enqueue_read_to_userbuffer_local);
HPX_OPENCL_TEMPLATE_ACTION_USES_MEDIUM_STACK(buffer,
                                             enqueue_read_to_userbuffer_remote);
HPX_OPENCL_TEMPLATE_ACTION_USES_MEDIUM_STACK(
    buffer, enqueue_read_to_userbuffer_rect_local);
HPX_OPENCL_TEMPLATE_ACTION_USES_MEDIUM_STACK(
    buffer, enqueue_read_to_userbuffer_rect_remote);
//]

////////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATIONS
//

// HPXCL tools
#include "../tools.hpp"

// other hpxcl dependencies
#include "util/event_dependencies.hpp"
#include "device.hpp"

template <typename T>
void hpx::opencl::server::buffer::enqueue_write(
    hpx::naming::id_type &&event_gid, std::size_t offset,
    hpx::serialization::serialize_buffer<T> data,
    std::vector<hpx::naming::id_type> &&dependencies) {
  HPX_ASSERT(hpx::opencl::tools::runs_on_medium_stack());

  cl_int err;
  cl_event return_event;

  // retrieve the dependency cl_events
  util::event_dependencies events(dependencies, parent_device.get());

  // retrieve the command queue
  cl_command_queue command_queue = parent_device->get_write_command_queue();

  // run the OpenCL-call
  err = clEnqueueWriteBuffer(command_queue, device_mem, CL_FALSE, offset,
                             data.size() * sizeof(T), data.data(),
                             static_cast<cl_uint>(events.size()),
                             events.get_cl_events(), &return_event);
  cl_ensure(err, "clEnqueueWriteBuffer()");

  // register the data to prevent deallocation
  parent_device->put_event_data(return_event, data);

  // register the cl_event to the client event
  parent_device->register_event(event_gid, return_event);
}

template <typename T>
void hpx::opencl::server::buffer::enqueue_write_rect(
    hpx::naming::id_type &&event_gid, hpx::opencl::rect_props &&rect_properties,
    hpx::serialization::serialize_buffer<T> data,
    std::vector<hpx::naming::id_type> &&dependencies) {
  HPX_ASSERT(hpx::opencl::tools::runs_on_medium_stack());

  cl_int err;
  cl_event return_event;

  // retrieve the dependency cl_events
  util::event_dependencies events(dependencies, parent_device.get());

  // retrieve the command queue
  cl_command_queue command_queue = parent_device->get_write_command_queue();

  // prepare arguments for OpenCL call
  std::size_t host_origin[] = {rect_properties.src_x * sizeof(T),
                               rect_properties.src_y, rect_properties.src_z};
  std::size_t buffer_origin[] = {rect_properties.dst_x * sizeof(T),
                                 rect_properties.dst_y, rect_properties.dst_z};
  std::size_t region[] = {rect_properties.size_x * sizeof(T),
                          rect_properties.size_y, rect_properties.size_z};

  HPX_ASSERT(data.size() >
             (rect_properties.size_x + rect_properties.src_x - 1) +
                 (rect_properties.size_y + rect_properties.src_y - 1) *
                     rect_properties.src_stride_y +
                 (rect_properties.size_z + rect_properties.src_z - 1) *
                     rect_properties.src_stride_z);

  // run the OpenCL-call
  err = clEnqueueWriteBufferRect(
      command_queue, device_mem, CL_FALSE, buffer_origin, host_origin, region,
      rect_properties.dst_stride_y * sizeof(T),
      rect_properties.dst_stride_z * sizeof(T),
      rect_properties.src_stride_y * sizeof(T),
      rect_properties.src_stride_z * sizeof(T), data.data(),
      static_cast<cl_uint>(events.size()), events.get_cl_events(),
      &return_event);
  cl_ensure(err, "clEnqueueWriteBufferRect()");

  // register the data to prevent deallocation
  parent_device->put_event_data(return_event, data);

  // register the cl_event to the client event
  parent_device->register_event(event_gid, return_event);
}

template <typename T>
void hpx::opencl::server::buffer::enqueue_read_to_userbuffer_local(
    hpx::naming::id_type &&event_gid, std::size_t offset,
    hpx::serialization::serialize_buffer<T> data,
    std::vector<hpx::naming::id_type> &&dependencies) {
  HPX_ASSERT(hpx::opencl::tools::runs_on_medium_stack());

  cl_int err;
  cl_event return_event;

  // retrieve the dependency cl_events
  util::event_dependencies events(dependencies, parent_device.get());

  // retrieve the command queue
  cl_command_queue command_queue = parent_device->get_read_command_queue();

  // run the OpenCL-call
  err = clEnqueueReadBuffer(command_queue, device_mem, CL_FALSE, offset,
                            data.size() * sizeof(T), data.data(),
                            static_cast<cl_uint>(events.size()),
                            events.get_cl_events(), &return_event);
  cl_ensure(err, "clEnqueueReadBuffer()");

  // register the data to prevent deallocation
  parent_device->put_event_data(return_event, data);

  // register the cl_event to the client event
  parent_device->register_event(event_gid, return_event);

  // arm the future. ! this blocks.
  parent_device->activate_deferred_event_with_data(event_gid);
}

template <typename T>
void hpx::opencl::server::buffer::enqueue_read_to_userbuffer_remote(
    hpx::naming::id_type &&event_gid, std::size_t offset, std::size_t size,
    std::uintptr_t remote_data_addr,
    std::vector<hpx::naming::id_type> &&dependencies) {
  HPX_ASSERT(hpx::opencl::tools::runs_on_medium_stack());

  typedef hpx::serialization::serialize_buffer<char> buffer_type;

  cl_int err;
  cl_event return_event;

  // retrieve the dependency cl_events
  util::event_dependencies events(dependencies, parent_device.get());

  // retrieve the command queue
  cl_command_queue command_queue = parent_device->get_read_command_queue();

  // create new target buffer
  buffer_type data(size);

  // run the OpenCL-call
  err = clEnqueueReadBuffer(command_queue, device_mem, CL_FALSE, offset,
                            data.size(), data.data(),
                            static_cast<cl_uint>(events.size()),
                            events.get_cl_events(), &return_event);
  cl_ensure(err, "clEnqueueReadBuffer()");

  // put_event_data not necessary as we locally keep the buffer alive until
  // the event triggered

  // also important: the cl_event does not get destroyed inside of
  // the event map of parent_device, because we keep the lcos::event
  // alive as we have an event_id

  // register the cl_event to the client event
  parent_device->register_event(event_gid, return_event);

  // prepare a zero-copy buffer
  hpx::opencl::lcos::zerocopy_buffer zerocopy_buffer(remote_data_addr, size,
                                                     data);

  // wait for the event to finish
  parent_device->wait_for_cl_event(return_event);

  // send the zerocopy_buffer to the lcos::event
  //     typedef hpx::opencl::lcos::detail::set_zerocopy_data_action<T>
  //         set_data_func;
  //     hpx::apply_colocated<set_data_func>(event_gid, event_gid,
  //     zerocopy_buffer);

  hpx::set_lco_value(event_gid, std::move(zerocopy_buffer));
}

template <typename T>
void hpx::opencl::server::buffer::enqueue_read_to_userbuffer_rect_local(
    hpx::naming::id_type &&event_gid, hpx::opencl::rect_props &&rect_properties,
    hpx::serialization::serialize_buffer<T> data,
    std::vector<hpx::naming::id_type> &&dependencies) {
  HPX_ASSERT(hpx::opencl::tools::runs_on_medium_stack());

  cl_int err;
  cl_event return_event;

  // retrieve the dependency cl_events
  util::event_dependencies events(dependencies, parent_device.get());

  // retrieve the command queue
  cl_command_queue command_queue = parent_device->get_read_command_queue();

  // prepare arguments for OpenCL call
  std::size_t buffer_origin[] = {rect_properties.src_x * sizeof(T),
                                 rect_properties.src_y, rect_properties.src_z};
  std::size_t host_origin[] = {rect_properties.dst_x * sizeof(T),
                               rect_properties.dst_y, rect_properties.dst_z};
  std::size_t region[] = {rect_properties.size_x * sizeof(T),
                          rect_properties.size_y, rect_properties.size_z};

  HPX_ASSERT(data.size() >
             (rect_properties.size_x + rect_properties.dst_x - 1) +
                 (rect_properties.size_y + rect_properties.dst_y - 1) *
                     rect_properties.dst_stride_y +
                 (rect_properties.size_z + rect_properties.dst_z - 1) *
                     rect_properties.dst_stride_z);

  // run the OpenCL-call
  err = clEnqueueReadBufferRect(
      command_queue, device_mem, CL_FALSE, buffer_origin, host_origin, region,
      rect_properties.src_stride_y * sizeof(T),
      rect_properties.src_stride_z * sizeof(T),
      rect_properties.dst_stride_y * sizeof(T),
      rect_properties.dst_stride_z * sizeof(T), data.data(),
      static_cast<cl_uint>(events.size()), events.get_cl_events(),
      &return_event);
  cl_ensure(err, "clEnqueueReadBufferRect()");

  // register the data to prevent deallocation
  parent_device->put_event_data(return_event, data);

  // register the cl_event to the client event
  parent_device->register_event(event_gid, return_event);

  // arm the future. ! this blocks.
  parent_device->activate_deferred_event_with_data(event_gid);
}

template <typename T>
void hpx::opencl::server::buffer::enqueue_read_to_userbuffer_rect_remote(
    hpx::naming::id_type &&event_gid, hpx::opencl::rect_props &&rect_properties,
    std::uintptr_t remote_data_addr,
    std::vector<hpx::naming::id_type> &&dependencies) {
  // the general algorithm of the remote rect read is:
  // - allocate a buffer that exactly fits the read data
  // - read from gpu
  // - send the data and extract it to the correct position in the
  //   remote destination buffer via zero-copy send

  HPX_ASSERT(hpx::opencl::tools::runs_on_medium_stack());

  typedef hpx::serialization::serialize_buffer<char> buffer_type;

  cl_int err;
  cl_event return_event;

  // retrieve the dependency cl_events
  util::event_dependencies events(dependencies, parent_device.get());

  // retrieve the command queue
  cl_command_queue command_queue = parent_device->get_read_command_queue();

  // create new target buffer
  std::size_t dst_size = rect_properties.size_x * rect_properties.size_y *
                         rect_properties.size_z * sizeof(T);
  buffer_type data(dst_size);

  // prepare arguments for OpenCL call
  std::size_t buffer_origin[] = {rect_properties.src_x * sizeof(T),
                                 rect_properties.src_y, rect_properties.src_z};
  std::size_t host_origin[] = {0, 0, 0};  // don't waste space on the host buf
  std::size_t region[] = {rect_properties.size_x * sizeof(T),
                          rect_properties.size_y, rect_properties.size_z};

  // run the OpenCL-call
  err = clEnqueueReadBufferRect(
      command_queue, device_mem, CL_FALSE, buffer_origin, host_origin, region,
      rect_properties.src_stride_y * sizeof(T),
      rect_properties.src_stride_z * sizeof(T),
      rect_properties.size_x * sizeof(T),
      rect_properties.size_x * sizeof(T) * rect_properties.size_y, data.data(),
      static_cast<cl_uint>(events.size()), events.get_cl_events(),
      &return_event);
  cl_ensure(err, "clEnqueueReadBufferRect()");

  // put_event_data not necessary as we locally keep the buffer alive until
  // the event triggered

  // also important: the cl_event does not get destroyed inside of
  // the event map of parent_device, because we keep the lcos::event
  // alive as we have an event_id

  // register the cl_event to the client event
  parent_device->register_event(event_gid, return_event);

  // prepare a zero-copy buffer
  // TODO replace dst_size with rect_properties
  hpx::opencl::lcos::zerocopy_buffer zerocopy_buffer(
      remote_data_addr, rect_properties, sizeof(T), data);

  // wait for the event to finish
  parent_device->wait_for_cl_event(return_event);

  // send the zerocopy_buffer to the lcos::event
  //     typedef hpx::opencl::lcos::detail::set_zerocopy_data_action<T>
  //         set_data_func;
  //     hpx::apply_colocated<set_data_func>(event_gid, event_gid,
  //     zerocopy_buffer);

  hpx::set_lco_value(event_gid, std::move(zerocopy_buffer));
}

#endif
