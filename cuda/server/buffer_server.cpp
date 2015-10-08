// Copyright (c)		2013 Damond Howard
//						2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include <hpx/hpx.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda/buffer.hpp"

namespace hpx {
namespace cuda {
namespace server {

buffer::buffer() {
}

buffer::buffer(size_t size, int parent_device_num) {
	this->parent_device_num = parent_device_num;
	this->arg_buffer_size = size;


	cudaStreamCreate( &stream );
	checkCudaError("buffer:buffer Create buffer's stream");
	cudaSetDevice(this->parent_device_num);
	checkCudaError("buffer:enqueue_read Set device");
	cudaMalloc((void**)&data_device, size);
	checkCudaError(
			"device::create_buffer Error during allocation of the device pointer");
	cudaMallocHost((void**)&data_host, size);
	checkCudaError(
			"device::create_buffer Error during allocation of the host pointer");
}

buffer::~buffer() {

	cudaSetDevice(this->parent_device_num);
	checkCudaError("buffer:enqueue_read Set device");
	cudaFree(this->data_device);
	checkCudaError("buffer::~buffer Error during free of the device pointer");
	cudaFreeHost(this->data_host);
	checkCudaError("buffer::~buffer Error during free of the device pointer");
	cudaStreamDestroy( stream );
	checkCudaError("buffer::~buffer Error during destroying of the stream");
}

size_t buffer::size() {
	return this->arg_buffer_size;
}

void buffer::set_size(size_t size) {
	this->arg_buffer_size = size;
}

hpx::serialization::serialize_buffer<char> buffer::enqueue_read(size_t offset,
		size_t size) {

	cudaSetDevice(this->parent_device_num);
	checkCudaError("buffer:enqueue_read Set device");

	cudaMemcpyAsync(this->data_host,this->data_device,this->arg_buffer_size,cudaMemcpyDeviceToHost,this->stream);
	checkCudaError(
			"buffer::enque_read Error during copy data from the device to the host");
	cudaStreamSynchronize( this->stream );
	checkCudaError(
				"buffer::enque_read Error during synchronization of stream");
	hpx::serialization::serialize_buffer<char> serializable_data(
			(char*) reinterpret_cast<char*>(this->data_host), size,
			hpx::serialization::serialize_buffer<char>::init_mode::reference);

	return serializable_data;

}

void buffer::enqueue_write(size_t offset,
		hpx::serialization::serialize_buffer<char> data) {

	cudaSetDevice(this->parent_device_num);
	checkCudaError("buffer:enqueue_read Set device");

	cudaMemcpyAsync(data_device,(void*)data.data(),this->arg_buffer_size,cudaMemcpyHostToDevice);
	checkCudaError(
			"buffer::enque_write Error during copy data from the host to the device");
	cudaStreamSynchronize( this->stream );
	checkCudaError(
				"buffer::enque_write Error during synchronization of stream");
}

void* buffer::get_raw_pointer() {
	return &this->data_device;
}

}
}
}

