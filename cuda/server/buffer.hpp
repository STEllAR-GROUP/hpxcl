// Copyright (c)     2013 Damond Howard
//                   2015 Patrick Diehl
//                   2017 Madhavan Seshadri
//                   2021 Pedro Barbosa
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#pragma once
#ifndef HPX_CUDA_SERVER_BUFFER_HPP_
#define HPX_CUDA_SERVER_BUFFER_HPP_

#include <hpx/hpx.hpp>

#include <cuda.h>
#include <cstdint>

#include "cuda/fwd_declarations.hpp"
#include "cuda/export_definitions.hpp"
#include "cuda/cuda_error_handling.hpp"
namespace hpx {
namespace cuda {
namespace server {
//////////////////////////////////////////////////////////
/// This class represents a buffer of cuda kernel arguments

class HPX_CUDA_EXPORT buffer
    : public hpx::components::locking_hook<
          hpx::components::managed_component_base<buffer> > {
 private:
  size_t arg_buffer_size;
  int parent_device_num;
  void* data_device;

// New stream if defined
#ifdef HPXCL_CUDA_WITH_STREAMS
  cudaStream_t stream;
#endif

 public:
  buffer();

  buffer(size_t size, int parent_device_num);

  size_t size();

  void set_size(size_t size);

  ~buffer();

  hpx::serialization::serialize_buffer<char> enqueue_read(size_t offset,
                                                          size_t size);

  hpx::serialization::serialize_buffer<char> enqueue_read_parcel(size_t offset,
                                                                 size_t size);

  uintptr_t enqueue_read_local(size_t offset, size_t size);

  uintptr_t enqueue_read_local_parcel(size_t offset, size_t size);

  void enqueue_write(size_t offset, size_t size,
                     hpx::serialization::serialize_buffer<char> data);

  void enqueue_write_parcel(size_t dst_offset, size_t size,
                            hpx::serialization::serialize_buffer<char> data, size_t src_offset);

  void enqueue_write_local(size_t offset, size_t size, uintptr_t data);

  void enqueue_write_local_parcel(size_t dst_offset, size_t size, uintptr_t data, size_t src_offset);

  uintptr_t get_device_pointer();

  void* get_raw_pointer();

  int get_device_id();

  void p2p_copy(uintptr_t dst, size_t dst_parent_device_id, size_t count);

#ifdef HPXCL_CUDA_WITH_STREAMS
  cudaStream_t get_stream();
#endif

  HPX_DEFINE_COMPONENT_ACTION(buffer, size);
  HPX_DEFINE_COMPONENT_ACTION(buffer, set_size);
  HPX_DEFINE_COMPONENT_ACTION(buffer, enqueue_read);
  HPX_DEFINE_COMPONENT_ACTION(buffer, enqueue_read_parcel);
  HPX_DEFINE_COMPONENT_ACTION(buffer, enqueue_read_local);
  HPX_DEFINE_COMPONENT_ACTION(buffer, enqueue_read_local_parcel);
  HPX_DEFINE_COMPONENT_ACTION(buffer, enqueue_write);
  HPX_DEFINE_COMPONENT_ACTION(buffer, enqueue_write_parcel);
  HPX_DEFINE_COMPONENT_ACTION(buffer, enqueue_write_local);
  HPX_DEFINE_COMPONENT_ACTION(buffer, enqueue_write_local_parcel);
  HPX_DEFINE_COMPONENT_ACTION(buffer, get_device_pointer);
  HPX_DEFINE_COMPONENT_ACTION(buffer, get_device_id);
  HPX_DEFINE_COMPONENT_ACTION(buffer, p2p_copy);
};
}  // namespace server
}  // namespace cuda
}  // namespace hpx

HPX_REGISTER_ACTION_DECLARATION(hpx::cuda::server::buffer::size_action,
                                buffer_size_action);
HPX_REGISTER_ACTION_DECLARATION(hpx::cuda::server::buffer::set_size_action,
                                buffer_set_size_action);
HPX_REGISTER_ACTION_DECLARATION(hpx::cuda::server::buffer::enqueue_read_action,
                                buffer_enqueue_read_action);
HPX_REGISTER_ACTION_DECLARATION(hpx::cuda::server::buffer::enqueue_read_parcel_action,
                                buffer_enqueue_read_parcel_action);
HPX_REGISTER_ACTION_DECLARATION(hpx::cuda::server::buffer::enqueue_write_action,
                                buffer_enqueue_write_action);
HPX_REGISTER_ACTION_DECLARATION(hpx::cuda::server::buffer::enqueue_write_parcel_action,
                                buffer_enqueue_write_parcel_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::buffer::enqueue_write_local_action,
    buffer_enqueue_write_local_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::buffer::enqueue_write_local_parcel_action,
    buffer_enqueue_write_local_parcel_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::buffer::enqueue_read_local_action,
    buffer_enqueue_read_local_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::buffer::enqueue_read_local_parcel_action,
    buffer_enqueue_read_local_parcel_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::buffer::get_device_pointer_action,
    buffer_get_device_pointer_action);
HPX_REGISTER_ACTION_DECLARATION(hpx::cuda::server::buffer::get_device_id_action,
                                buffer_get_device_id_action);
HPX_REGISTER_ACTION_DECLARATION(hpx::cuda::server::buffer::p2p_copy_action,
                                buffer_p2p_copy_action);

#endif  // BUFFER_2_HPP
