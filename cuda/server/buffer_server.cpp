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

	cudaMemcpy(this->data_host,this->data_device,this->arg_buffer_size,cudaMemcpyDeviceToHost);

	checkCudaError(
			"buffer::enque_read Error during copy data from the device to the host");
	//std::cout << "Result in enqueue read " << ((int*)this->data_host)[0] << std::endl;
	hpx::serialization::serialize_buffer<char> serializable_data(
			(char*) reinterpret_cast<char*>(this->data_host), size,
			hpx::serialization::serialize_buffer<char>::init_mode::reference);

	return serializable_data;

}

void buffer::enqueue_write(size_t offset,
		hpx::serialization::serialize_buffer<char> data) {

	cudaSetDevice(this->parent_device_num);
	checkCudaError("buffer:enqueue_read Set device");



	//std::cout << "Data Server: " << data.data()[0] << " " << data.data() << " " << this->arg_buffer_size << std::endl;

  //  int* testData;
//	cudaMallocHost((void**)&testData,this->arg_buffer_size);

//	for (unsigned int i = 0; i< this->arg_buffer_size / 4 ; i++){
//		testData[i] = 1;
		//std::cout << i << " " <<  data.data()[i] << std::endl;
//	}



	cudaMemcpy(data_device,(void*)data.data(),this->arg_buffer_size,cudaMemcpyHostToDevice);
	checkCudaError(
			"buffer::enque_write Error during copy data from the host to the device");
	//cudaFreeHost(testData);

}

void* buffer::get_raw_pointer() {
	return &this->data_device;
}

}
}
}

