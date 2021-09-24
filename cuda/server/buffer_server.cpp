// Copyright (c)		2013 Damond Howard
//						2015 Patrick Diehl
//						2017 Madhavan Seshadri
//						2021 Pedro Barbosa
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include <hpx/hpx.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda/buffer.hpp"

namespace hpx {
namespace cuda {
namespace server {

/**
 * Default constructor
 */
buffer::buffer() {}

buffer::buffer(size_t size, int parent_device_num) {
  this->parent_device_num = parent_device_num;
  this->arg_buffer_size = size;

  // Set the CUDA device
  cudaSetDevice(this->parent_device_num);
  checkCudaError("buffer:enqueue_read Set device");

#ifdef HPXCL_CUDA_WITH_STREAMS
  cudaStreamCreate(&stream);
  checkCudaError("buffer:buffer Create buffer's stream");
#endif
  cudaMalloc((void**)&data_device, size);
  checkCudaError(
      "buffer::buffer Error during allocation of the device pointer");
}

/**
 * Default destructor
 */
buffer::~buffer() {
  cudaSetDevice(this->parent_device_num);
  checkCudaError("buffer::~buffer Error in setting device");

  // Synchronize the stream so that all operations are completed before stream
  // is destroyed
#ifdef HPXCL_CUDA_WITH_STREAMS
  cudaStreamSynchronize(this->stream);
  checkCudaError("buffer::~buffer Error during synchronization of stream");
#endif

  cudaFree(data_device);
  checkCudaError("buffer::~buffer Error during free of the device pointer");

  // Destroy the buffer stream created
#ifdef HPXCL_CUDA_WITH_STREAMS
  cudaStreamDestroy(stream);
  checkCudaError("buffer::~buffer Error during destroying of the stream");
#endif
}

/**
 * Returns the size of the buffer
 */
size_t buffer::size() { return this->arg_buffer_size; }

/**
 * Set the size of the buffer
 */
void buffer::set_size(size_t size) { this->arg_buffer_size = size; }

hpx::serialization::serialize_buffer<char> buffer::enqueue_read(size_t offset,
                                                                size_t size) {
  size_t localSize = this->arg_buffer_size - offset * size;
  void* data_host;
  cudaSetDevice(this->parent_device_num);
  checkCudaError("buffer:enqueue_read Set device");
  cudaMallocHost((void**)&data_host, localSize);
  checkCudaError("buffer:enqueue_read allocate host memory");
  char* slicedPointer = (char*)(this->data_device) + offset;

// Asynchronous copy from device to Host call
#ifdef HPXCL_CUDA_WITH_STREAMS
  cudaMemcpyAsync(data_host, (void*)slicedPointer, localSize,
                  cudaMemcpyDeviceToHost, this->stream);
  checkCudaError(
      "buffer::enque_read Error during copy data from the device to the host");
  cudaStreamSynchronize(this->stream);
  checkCudaError("buffer::enque_read Error during synchronization of stream");
#else
  cudaMemcpyAsync(data_host, (void*)slicedPointer, localSize,
                  cudaMemcpyDeviceToHost);
  checkCudaError(
      "buffer::enque_read Error during copy data from the device to the host");
#endif
  hpx::serialization::serialize_buffer<char> serializable_data(
      (char*)reinterpret_cast<char*>(data_host), size,
      hpx::serialization::serialize_buffer<char>::init_mode::reference);

  return serializable_data;
}

hpx::serialization::serialize_buffer<char> buffer::enqueue_read_parcel(size_t offset,
                                                                size_t size) {
  void* data_host;
  cudaSetDevice(this->parent_device_num);
  checkCudaError("buffer:enqueue_read Set device");
  cudaMallocHost((void**)&data_host, size);
  checkCudaError("buffer:enqueue_read allocate host memory");
  char* slicedPointer = (char*)(this->data_device) + offset;

// Asynchronous copy from device to Host call
#ifdef HPXCL_CUDA_WITH_STREAMS
  cudaMemcpyAsync(data_host, (void*)slicedPointer, size,
                  cudaMemcpyDeviceToHost, this->stream);
  checkCudaError(
      "buffer::enque_read Error during copy data from the device to the host");
  cudaStreamSynchronize(this->stream);
  checkCudaError("buffer::enque_read Error during synchronization of stream");
#else
  cudaMemcpyAsync(data_host, (void*)slicedPointer, size,
                  cudaMemcpyDeviceToHost);
  checkCudaError(
      "buffer::enque_read Error during copy data from the device to the host");
#endif
  hpx::serialization::serialize_buffer<char> serializable_data(
      (char*)reinterpret_cast<char*>(data_host), size,
      hpx::serialization::serialize_buffer<char>::init_mode::reference);

  return serializable_data;
}

void buffer::enqueue_write(size_t offset, size_t size,
                           hpx::serialization::serialize_buffer<char> data) {
  cudaSetDevice(this->parent_device_num);
  checkCudaError("buffer:enqueue_read Set device");
  char* slicedPointer = (char*)(data.data()) + offset;

// Asynchronous copy from Host to device call -- Non-blocking on host
#ifdef HPXCL_CUDA_WITH_STREAMS
  cudaMemcpyAsync(this->data_device, (void*)slicedPointer, size,
                  cudaMemcpyHostToDevice, this->stream);
  checkCudaError(
      "buffer::enque_write Error during copy data from the host to the device");
  cudaStreamSynchronize(this->stream);
  checkCudaError("buffer::enque_read Error during synchronization of stream");
#else
  cudaMemcpyAsync(this->data_device, (void*)slicedPointer, size,
                  cudaMemcpyHostToDevice);
  checkCudaError(
      "buffer::enque_write Error during copy data from the host to the device");
#endif
}

/**
 * Get the device pointer
 */
uintptr_t buffer::get_device_pointer() {
  return reinterpret_cast<uintptr_t>(data_device);
}

void* buffer::get_raw_pointer() { return &data_device; }

/**
 * Get the device id
 */
int buffer::get_device_id() { return this->parent_device_num; }

void buffer::enqueue_write_local(size_t offset, size_t size, uintptr_t data) {
  cudaSetDevice(this->parent_device_num);
  checkCudaError("buffer:enqueue_read Set device");
  char* slicedPointer = reinterpret_cast<char*>(data) + offset;

// Asynchronous copy from Host to device call -- Non-blocking on host
#ifdef HPXCL_CUDA_WITH_STREAMS
  cudaMemcpyAsync(this->data_device, (void*)slicedPointer, size,
                  cudaMemcpyHostToDevice, this->stream);
  checkCudaError(
      "buffer::enque_write Error during copy data from the host to the device");
  cudaStreamSynchronize(this->stream);
  checkCudaError("buffer::enque_read Error during synchronization of stream");
#else
  cudaMemcpyAsync(this->data_device, (void*)slicedPointer, size,
                  cudaMemcpyHostToDevice);
  checkCudaError(
      "buffer::enque_write Error during copy data from the host to the device");
#endif
}

uintptr_t buffer::enqueue_read_local(size_t offset, size_t size) {
  size_t localSize = this->arg_buffer_size - offset * size;
  void* data_host;
  cudaSetDevice(this->parent_device_num);
  checkCudaError("buffer:enqueue_read Set device");
  cudaMallocHost((void**)&data_host, localSize);
  checkCudaError("buffer:enqueue_read allocate host memory");
  char* slicedPointer = (char*)(this->data_device) + offset;

// Asynchronous copy from device to host
#ifdef HPXCL_CUDA_WITH_STREAMS
  cudaMemcpyAsync(data_host, (void*)slicedPointer, localSize,
                  cudaMemcpyDeviceToHost, this->stream);
  checkCudaError(
      "buffer::enque_read Error during copy data from the device to the host");
  cudaStreamSynchronize(this->stream);
  checkCudaError("buffer::enque_read Error during synchronization of stream");
#else
  cudaMemcpyAsync(data_host, (void*)slicedPointer, localSize,
                  cudaMemcpyDeviceToHost);
  checkCudaError(
      "buffer::enque_read Error during copy data from the device to the host");
#endif
  return reinterpret_cast<uintptr_t>(data_host);
}

uintptr_t buffer::enqueue_read_local_parcel(size_t offset, size_t size) {
  void* data_host;
  cudaSetDevice(this->parent_device_num);
  checkCudaError("buffer:enqueue_read Set device");
  cudaMallocHost((void**)&data_host, size);
  checkCudaError("buffer:enqueue_read allocate host memory");
  char* slicedPointer = (char*)(this->data_device) + offset;

// Asynchronous copy from device to host
#ifdef HPXCL_CUDA_WITH_STREAMS
  cudaMemcpyAsync(data_host, (void*)slicedPointer, size,
                  cudaMemcpyDeviceToHost, this->stream);
  checkCudaError(
      "buffer::enque_read Error during copy data from the device to the host");
  cudaStreamSynchronize(this->stream);
  checkCudaError("buffer::enque_read Error during synchronization of stream");
#else
  cudaMemcpyAsync(data_host, (void*)slicedPointer, size,
                  cudaMemcpyDeviceToHost);
  checkCudaError(
      "buffer::enque_read Error during copy data from the device to the host");
#endif
  return reinterpret_cast<uintptr_t>(data_host);
}

void buffer::p2p_copy(uintptr_t dst, size_t dst_parent_device_id,
                      size_t count) {
  int can_peer;

  cudaDeviceCanAccessPeer(&can_peer, this->parent_device_num,
                          dst_parent_device_id);
  checkCudaError("buffer::p2p Error checking peer access");

  if (can_peer == 0) {
    std::cerr << "Error: P2P copy between these devices is not possible!"
              << std::endl;

  } else {
    cudaSetDevice(dst_parent_device_id);
    checkCudaError("buffer::p2p_copy Set dest device");
    cudaDeviceEnablePeerAccess(this->parent_device_num, 0);
    checkCudaError("buffer::p2p_copy Enable p2p on the parent device");
    cudaSetDevice(this->parent_device_num);
    checkCudaError("buffer::p2p_copy Set source device");
    cudaDeviceEnablePeerAccess(dst_parent_device_id, 0);
    checkCudaError("buffer::p2p_copy Enable p2p in the source device");

#ifdef HPXCL_CUDA_WITH_STREAMS
    cudaMemcpyPeerAsync((void*)dst, dst_parent_device_id, this->data_device,
                        this->parent_device_num, count, this->stream);
    checkCudaError("buffer::p2p_copy Error during copy data between devices");
    cudaStreamSynchronize(this->stream);
    checkCudaError("buffer::p2p_copy Error during synchronization of stream");

#else
    cudaMemcpyPeerAsync((void*)dst, dst_parent_device_id, this->data_device,
                        this->parent_device_num, count);
    checkCudaError("buffer::p2p_copy Error during copy data between devices");
#endif
  }
}

#ifdef HPXCL_CUDA_WITH_STREAMS
cudaStream_t buffer::get_stream() { return this->stream; }
#endif

}  // namespace server
}  // namespace cuda
}  // namespace hpx
