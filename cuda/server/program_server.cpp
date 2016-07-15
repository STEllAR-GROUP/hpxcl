// Copyright (c)    2013 Damond Howard
//                  2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include <hpx/hpx.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda/server/program.hpp"
#include "cuda/server/buffer.hpp"

#include <sstream>
#include <string>

namespace hpx {
namespace cuda {
namespace server {

program::program() {

}

program::program(int parent_device_id) {
	this->parent_device_id = parent_device_id;
	cudaSetDevice(parent_device_id);
	checkCudaError("program::program Error setting the device");
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	checkCudaError("program::program Error in creating default stream");
	this->streams.push_back(stream);
}

program::program(hpx::naming::id_type device_id, std::string code) {
}

program::program(hpx::naming::id_type device_id,
		hpx::serialization::serialize_buffer<char> binary) {
}

program::~program() {

	cudaSetDevice(this->parent_device_id);
	nvrtcDestroyProgram(&prog);
	checkCudaError("program::~program Destroy Program");

	for (auto stream : streams) {
		cudaStreamDestroy(stream);
		checkCudaError("program::~program Destroy stream");
	}

	cuModuleUnload(module);
	checkCudaError("program::~programDestroy module");
}

void program::set_source(std::string source) {
	this->kernel_source = source;
}

void program::build(std::vector<std::string> compilerFlags,
		std::vector<std::string> modulenames, unsigned int debug) {

	cudaSetDevice(this->parent_device_id);
	boost::uuids::uuid uuid = boost::uuids::random_generator()();
	std::string filename = to_string(uuid);
	filename.append(".cu");

	if (debug == 1) {
		compilerFlags.push_back("-G");
		compilerFlags.push_back("-lineinfo");
	}

	nvrtcCreateProgram(&prog, this->kernel_source.c_str(), filename.c_str(), 0,
			NULL, NULL);
	checkCudaError("program::build Create Program");
	const char * opts[compilerFlags.size()];
	unsigned int i = 0;
	for (auto opt : compilerFlags) {
		opts[i] = compilerFlags[i].c_str();
		i++;
	}

	nvrtcResult compileResult = nvrtcCompileProgram(prog, compilerFlags.size(),
			opts);

	if (compileResult != NVRTC_SUCCESS) {
		size_t logSize;
		nvrtcGetProgramLogSize(prog, &logSize);
		checkCudaError("program::build Create Log");
		char *log = new char[logSize];
		nvrtcGetProgramLog(prog, log);
		checkCudaError("program::build Get Log");

		std::cout << log << std::endl;
		delete[] log;
		exit(1);
	}

	size_t ptxSize;

	nvrtcGetPTXSize(prog, &ptxSize);
	checkCudaError("program::build Get ptx size");

	char *ptx = new char[ptxSize];
	nvrtcGetPTX(prog, ptx);
	checkCudaError("program::build Get ptx of Program");

	cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
	checkCudaError("program::build Load Module");

	for (auto modulename : modulenames) {

		CUfunction kernel;
		cuModuleGetFunction(&kernel, module, modulename.c_str());
		checkCudaError("program::build Get Function");
		kernels.insert(std::pair<std::string, CUfunction>(modulename, kernel));
	}

}

void program::run(std::vector<hpx::naming::id_type> args,
		std::string modulename, Dim3 grid, Dim3 block,
		std::vector<hpx::naming::id_type> dependencies, int stream) {

	std::vector<void*> args_pointer(args.size());

	//void *args_pointer[args.size()];

	unsigned int i = 0;
	for (auto arg : args) {
		auto buffer = hpx::get_ptr<hpx::cuda::server::buffer>(arg).get();
		void* tmp = buffer->get_raw_pointer();
		args_pointer[i] = tmp;
		i++;
	}

	cudaSetDevice(this->parent_device_id);
	checkCudaError("program::run Error setting the device");

	//Run on the default stream
	if (dependencies.size() == 0 and stream == -1) {
		cuLaunchKernel(this->kernels[modulename], grid.x, grid.y, grid.y,
				block.x, block.y, block.z, 0, this->streams[0], args_pointer.data(),
				0);
		checkCudaError("program::run Run kernel");
		cudaStreamSynchronize(this->streams[0]);
		checkCudaError("program::run Error during synchronization of stream");
	}
	//Run on the provided stream
	else if (dependencies.size() == 0 and stream >= 0) {
		cuLaunchKernel(this->kernels[modulename], grid.x, grid.y, grid.y,
				block.x, block.y, block.z, 0, this->streams[stream], // shared mem and stream
				args_pointer.data(), 0);
		checkCudaError("program::run Run kernel");
		cudaStreamSynchronize(this->streams[stream]);
		checkCudaError("program::run Error during synchronization of stream");
	}
	//Run on the one stream of the dependency
	else if (dependencies.size() == 1 and stream == -1) {
		auto buffer =
				hpx::get_ptr<hpx::cuda::server::buffer>(dependencies[0]).get();
		cuLaunchKernel(this->kernels[modulename], grid.x, grid.y, grid.y,
				block.x, block.y, block.z, 0, buffer->get_stream(), // shared mem and stream
				args_pointer.data(), 0);
		checkCudaError("program::run Run kernel");
		cudaStreamSynchronize(buffer->get_stream());
		checkCudaError("program::run Error during synchronization of stream");

	}
	//Run on the provided stream
	else if (dependencies.size() > 1 and stream >= 0) {
		for (auto dependency : dependencies) {
			auto buffer = hpx::get_ptr<hpx::cuda::server::buffer>(
					dependencies[0]).get();
			cudaStreamSynchronize(buffer->get_stream());
			checkCudaError(
					"program::run Error during synchronization of stream");
		}

		cuLaunchKernel(this->kernels[modulename], grid.x, grid.y, grid.y,
				block.x, block.y, block.z, 0, this->streams[stream], // shared mem and stream
				args_pointer.data(), 0);
		checkCudaError("program::run Run kernel");
		cudaStreamSynchronize(this->streams[stream]);
		checkCudaError("program::run Error during synchronization of stream");
	} else {
		for (auto dependency : dependencies) {
			auto buffer = hpx::get_ptr<hpx::cuda::server::buffer>(
					dependencies[0]).get();
			cudaStreamSynchronize(buffer->get_stream());
			checkCudaError(
					"program::run Error during synchronization of stream");
		}

		cuLaunchKernel(this->kernels[modulename], grid.x, grid.y, grid.y,
				block.x, block.y, block.z, 0, this->streams[0], // shared mem and stream
				args_pointer.data(), 0);
		checkCudaError("program::run Run kernel");
		cudaStreamSynchronize(this->streams[0]);
		checkCudaError("program::run Error during synchronization of stream");
	}

}

unsigned int program::get_streams_size() {
	return this->streams.size();
}

unsigned int program::create_stream() {
	cudaSetDevice(parent_device_id);
	checkCudaError("program::program Error setting the device");
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	checkCudaError("program::program Error in creating default stream");
	this->streams.push_back(stream);
	return this->streams.size() - 1;
}

}
}
}

