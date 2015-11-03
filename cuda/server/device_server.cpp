// Copyright (c)    2013 Damond Howard
//                  2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include <hpx/hpx.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include <boost/make_shared.hpp>

#include "cuda/server/device.hpp"


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

	cudaGetDeviceProperties(&props, device_id);
	checkCudaError("device::device");

	this->device_name = props.name;
}

device::~device() {
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

	cudaDeviceProp props;
	cudaError_t error;
	error = cudaGetDeviceProperties(&props, this->device_id);
	if (error == cudaErrorInvalidDevice) {
		std::cout << "Device does not exist" << std::endl;
	}

	std::cout << props.name << std::endl;
	std::cout << "   Global memory:   " << props.totalGlobalMem / mb << "mb"
			<< std::endl;
	std::cout << "   Shared memory:   " << props.sharedMemPerBlock / kb << "kb"
			<< std::endl;
	std::cout << "   Constant memory: " << props.totalConstMem / kb << "kb"
			<< std::endl;
	std::cout << "   Block registers: " << props.regsPerBlock << std::endl
			<< std::endl;
	std::cout << "   Warp size:         " << props.warpSize << std::endl;
	std::cout << "   Threads per block: " << props.maxThreadsPerBlock
			<< std::endl;
	std::cout << "   Max block dimensions: [ " << props.maxThreadsDim[0] << ", "
			<< props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]"
			<< std::endl;
	std::cout << "   Max grid dimensions:  [ " << props.maxGridSize[0] << ", "
			<< props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]"
			<< std::endl;
	std::cout << "   Multiprocessor Count: " << props.multiProcessorCount
			<< std::endl;
	std::cout << std::endl;
	std::cout << "   Unified addressing: " << props.unifiedAddressing
			<< std::endl;
	std::cout << "   Concurrent kernels: " << props.concurrentKernels
			<< std::endl;
	std::cout << "   Diver Overlap: " << props.deviceOverlap << std::endl;
	std::cout << "   Memory Clock Rate: " << props.memoryClockRate << std::endl;
	std::cout << "   Memory Bus Width: " << props.memoryBusWidth << std::endl;
	std::cout << "   l2 Cache Size: " << props.l2CacheSize << std::endl;
	std::cout << "   Clock Rate: " << props.clockRate << std::endl;
	std::cout << "   Exec Time Out: " << props.kernelExecTimeoutEnabled
			<< std::endl << std::endl;

	std::cout << "   Compute Capability: " << props.major << "." << props.minor
			<< std::endl;
	std::cout << "   Compute Modes: " << props.computeMode << std::endl
			<< std::endl;

	//}
}

void device::get_extended_cuda_info() {

	this->get_cuda_info();

	std::cout << "   Max Texture 1D: " << props.maxTexture1D << std::endl;
	std::cout << "   Max Texture 1D Linear: " << props.maxTexture1DLinear
			<< std::endl;
	this->print2D("Max Texture 2D", props.maxTexture2D);
	this->print3D("Max Texture 2D Linear", props.maxTexture2DLinear);
	this->print2D("Max Texture 2D Gather", props.maxTexture2DGather);
	this->print3D("Max Texture 3D", props.maxTexture3D);
	std::cout << "   Max Texture Cubemap: " << props.maxTextureCubemap
			<< std::endl;
	this->print2D("Max Texture 1D Layered", props.maxTexture1DLayered);
	this->print3D("Max Texture 2D Layered", props.maxTexture2DLayered);
	this->print2D("Max Texture Cubemap Layered",
			props.maxTextureCubemapLayered);
	std::cout << "   Max Surface 1D: " << props.maxSurface1D << std::endl;
	this->print2D("Max Surface 2D", props.maxSurface2D);
	this->print3D("Max Surface 3D", props.maxSurface3D);
	this->print2D("Max Surface 1D Layered", props.maxSurface1DLayered);
	this->print3D("Max Surface 2D layered", props.maxSurface2DLayered);
	std::cout << "   Max Surface Cubemap: " << props.maxSurfaceCubemap
			<< std::endl;
	this->print2D("Max Surface Cubemap Layered",
			props.maxSurfaceCubemapLayered);
	std::cout << "   Surface Alignment: " << props.surfaceAlignment
			<< std::endl;

}

void device::print2D(std::string name, int * array) {

	std::cout << "   " << name << ": [ " << array[0] << ", " << array[1] << " ]"
			<< std::endl;
}

void device::print3D(std::string name, int * array) {

	std::cout << "   " << name << ": [ " << array[0] << ", " << array[1] << ", "
			<< array[2] << " ]" << std::endl;
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
	boost::shared_ptr<hpx::lcos::local::promise<int> > p = boost::make_shared<
			hpx::lcos::local::promise<int> >();

	hpx::util::io_service_pool* pool = hpx::get_runtime().get_thread_pool(
			"io_pool");
	pool->get_io_service().post(hpx::util::bind(&do_wait, p));

	return p->get_future();
}


hpx::cuda::program device::create_program_with_source(std::string source) {
	typedef hpx::cuda::server::program program_type;

	hpx::cuda::program cu_program(
			hpx::components::new_ < program_type > (hpx::find_here(),this->device_id));
	cu_program.set_source_sync(source);
	return cu_program;
}

hpx::cuda::buffer device::create_buffer(size_t size) {
	typedef hpx::cuda::server::buffer buffer_type;

	hpx::cuda::buffer cu_buffer(
			hpx::components::new_ < buffer_type > (hpx::find_here(),size,this->device_id));

	return cu_buffer;
}

void
device::release_event(hpx::naming::gid_type gid)
{
    //HPX_ASSERT(hpx::opencl::tools::runs_on_large_stack());

    // release data registered on event
    //delete_event_data(event_map.get(gid));

    // delete event from map
    //event_map.remove(gid);

}

void
device::activate_deferred_event(hpx::naming::id_type event_id)
{
    // get the cl_event
    //cl_event event = event_map.get(event_id);

    // wait for the cl_event to complete
    //wait_for_cl_event(event);

    // trigger the client event
    //hpx::trigger_lco_event(event_id, false);

}


int device::get_device_architecture_major() {

	return this->props.major;
}

int device::get_device_architecture_minor() {

	return this->props.minor;
}
}
}
}
