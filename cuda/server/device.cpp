// Copyright (c)		2013 Damond Howard
//						2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

/*
 #include <hpx/hpx_fwd.hpp>
 #include <hpx/runtime/components/server/managed_component_base.hpp>
 #include <hpx/runtime/components/server/locking_hook.hpp>
 #include <hpx/runtime/actions/component_action.hpp>
 #include <hpx/include/util.hpp>
 #include <hpx/hpx_init.hpp>
 #include <hpx/include/runtime.hpp>
 */
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <thrust/version.h>
//#include <boost/make_shared.hpp>
//#include <string>
//#include <sstream>
//#include <iostream>
//#include <vector>
//#include "../cuda/kernel.cuh"
#include "device.hpp"

namespace hpx {
namespace cuda {
namespace server {

device::device() {
	cuInit(0);
	cuDeviceGet(&cu_device, 0);
	cuCtxCreate(&cu_context, 0, cu_device);
	device_name = props.name;
}

device::device(int device_id) {
	cuInit(0);
	cuDeviceGet(&cu_device, device_id);
	cuCtxCreate(&cu_context, 0, cu_device);
	this->set_device(device_id);
	cudaError_t error;
	error = cudaGetDeviceProperties(&props, device_id);
	this->device_name = props.name;
}

device::~device() {
}

void device::free() {
	for (uint64_t i = 0; i < device_ptrs.size(); i++) {
		cuMemFree(device_ptrs[i].device_ptr);
	}
	cuCtxDetach(cu_context);
}

int device::get_device_count() {
	int device_count = 0;
	cuDeviceGetCount(&device_count);
	return device_count;
}

void device::set_device(int dev) {
	this->device_id = dev;
	CUresult error;
	error = cuCtxSetCurrent(cu_context);
}

void device::get_cuda_info() {
	const int kb = 1024;
	const int mb = kb * kb;

	std::cout << "CUDA version:   v" << CUDART_VERSION << std::endl;
	std::cout << "Thrust version: v" << THRUST_MAJOR_VERSION << "."
			<< THRUST_MINOR_VERSION << std::endl << std::endl;

	int dev_count = this->get_device_count();

	if (dev_count <= 0) {
		std::cout << "No CUDA devices on the current locality" << std::endl;
	} else if (dev_count > 0) {
		std::cout << "CUDA Devices: " << std::endl << std::endl;
	}
	for (int i = 0; i < dev_count; ++i) {
		cudaDeviceProp props;
		cudaError_t error;
		error = cudaGetDeviceProperties(&props, i);
		if (error == cudaErrorInvalidDevice) {
			std::cout << "Device does not exist" << std::endl;
		}

		std::cout << i << ": " << props.name << ": " << props.major << "."
				<< props.minor << std::endl;
		std::cout << "   Global memory:   " << props.totalGlobalMem / mb << "mb"
				<< std::endl;
		std::cout << "   Shared memory:   " << props.sharedMemPerBlock / kb
				<< "kb" << std::endl;
		std::cout << "   Constant memory: " << props.totalConstMem / kb << "kb"
				<< std::endl;
		std::cout << "   Block registers: " << props.regsPerBlock << std::endl
				<< std::endl;
		std::cout << "   Warp size:         " << props.warpSize << std::endl;
		std::cout << "   Threads per block: " << props.maxThreadsPerBlock
				<< std::endl;
		std::cout << "   Max block dimensions: [ " << props.maxThreadsDim[0]
				<< ", " << props.maxThreadsDim[1] << ", "
				<< props.maxThreadsDim[2] << " ]" << std::endl;
		std::cout << "   Max grid dimensions:  [ " << props.maxGridSize[0]
				<< ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2]
				<< " ]" << std::endl;
		std::cout << std::endl;
	}
}

int device::get_device_id() {
	return this->device_id;
}

int device::get_context() {
	return this->context_id;
}

int device::get_all_devices() {
	int num_devices = get_device_count();
	return num_devices;
}

void device::do_wait(boost::shared_ptr<hpx::lcos::local::promise<int> > p) {
	p->set_value(0);
}

hpx::lcos::future<int> device::wait() {
	boost::shared_ptr < hpx::lcos::local::promise<int> > p = boost::make_shared<
			hpx::lcos::local::promise<int> >();

	hpx::util::io_service_pool* pool = hpx::get_runtime().get_thread_pool(
			"io_pool");
	pool->get_io_service().post(hpx::util::bind(&do_wait, p));

	return p->get_future();
}

void device::create_device_ptr(size_t const byte_count) {
	Device_ptr temp;
	cuMemAlloc(&temp.device_ptr, byte_count);
	temp.byte_count = byte_count;
	device_ptrs.push_back(temp);
	Host_ptr<int> temp2;
	temp2.host_ptr = (int*) malloc(byte_count);
	host_ptrs.push_back(temp2);
}

void device::mem_cpy_h_to_d(unsigned int variable_id) {
	cuMemcpyHtoD(device_ptrs[variable_id].device_ptr,
			host_ptrs[variable_id].host_ptr,
			device_ptrs[variable_id].byte_count);
}

void device::mem_cpy_d_to_h(unsigned int variable_id) {
	cuMemcpyDtoH(host_ptrs[variable_id].host_ptr,
			device_ptrs[variable_id].device_ptr,
			device_ptrs[variable_id].byte_count);
}

void device::launch_kernel(hpx::cuda::kernel cu_kernel) {
	hpx::cuda::server::kernel::Dim3 block = cu_kernel.get_block_sync();
	hpx::cuda::server::kernel::Dim3 grid = cu_kernel.get_grid_sync();

	void *args[1] = { &(device_ptrs[0].device_ptr) };

	CUfunction cu_function;
	CUmodule cu_module;
	CUresult cu_error;

	cu_error = cuModuleLoad(&cu_module,
			(char*) cu_kernel.get_module_sync().c_str());
	std::cout << "loading module returns " << (unsigned int) cu_error
			<< std::endl;

	cu_error = cuModuleGetFunction(&cu_function, cu_module,
			(char*) cu_kernel.get_function_sync().c_str());
	std::cout << "loading function returns " << (unsigned int) cu_error
			<< std::endl;

	cu_error = cuLaunchKernel(cu_function, grid.x, grid.y, grid.z, block.x,
			block.y, block.z, 0, 0, args, 0);
	std::cout << "launching kernel returns " << (unsigned int) cu_error
			<< std::endl;
}

hpx::cuda::program device::create_program_with_source(std::string source) {
	typedef hpx::cuda::server::program program_type;

	hpx::cuda::program cu_program(
			hpx::components::new_ < program_type > (hpx::find_here()));
	cu_program.set_source_sync(source);
	return cu_program;
}

hpx::cuda::buffer device::create_buffer(size_t size) {
	typedef hpx::cuda::server::buffer buffer_type;

	hpx::cuda::buffer cu_buffer(
			hpx::components::new_ < buffer_type > (hpx::find_here()));

	cu_buffer.set_size_sync(size);

	return cu_buffer;
}
}
}
}
