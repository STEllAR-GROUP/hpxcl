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

	nvrtcDestroyProgram(&prog);
	checkCudaError("Destroy Program");

	for (auto stream : streams) {
		cudaStreamDestroy(stream);
		checkCudaError("Destroy stream");
	}

	cuModuleUnload(module);
	checkCudaError("Destroy module");
}

void program::set_source(std::string source) {
	this->kernel_source = source;
}

//ToDo: Add debug flag
void program::build(std::vector<std::string> compilerFlags, std::string modulename,
		unsigned int debug) {

	boost::uuids::uuid uuid = boost::uuids::random_generator()();
	std::string filename = to_string(uuid);
	filename.append(".cu");

	if (debug == 1) {
		compilerFlags.push_back("-G");
		compilerFlags.push_back("-lineinfo");
	}

	nvrtcCreateProgram(&prog, this->kernel_source.c_str(),
			filename.c_str(), 0, NULL, NULL);
	checkCudaError("Create Program");
	const char * opts[compilerFlags.size()];
	unsigned int i = 0;
	for (auto opt : compilerFlags) {
		opts[i] = compilerFlags[i].c_str();
		i++;
	}


	nvrtcResult compileResult = nvrtcCompileProgram(prog,
			compilerFlags.size(), opts);

	if (compileResult != NVRTC_SUCCESS) {
		size_t logSize;
		nvrtcGetProgramLogSize(prog, &logSize);
		checkCudaError("Create Log");
		char *log = new char[logSize];
		nvrtcGetProgramLog(prog, log);
		checkCudaError("get Log");

		std::cout << log << std::endl;
		delete[] log;
		exit(1);
	}

	size_t ptxSize;

	nvrtcGetPTXSize(prog, &ptxSize);
	checkCudaError("Get ptx size");

	char *ptx = new char[ptxSize];
	nvrtcGetPTX(prog, ptx);
	checkCudaError("Get ptx of Program");

	cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
	checkCudaError("Load Module");
	cuModuleGetFunction(&(this->kernel), module, modulename.c_str());
	checkCudaError("Get Function");
}

void program::run(std::vector<hpx::naming::id_type> args,
		std::string modulename, Dim3 grid, Dim3 block, unsigned int stream) {

	void *args_pointer[args.size()];

	unsigned int i = 0;
	for (auto arg : args) {
		auto buffer = hpx::get_ptr<hpx::cuda::server::buffer>(arg).get();
		void* tmp = buffer->get_raw_pointer();
		args_pointer[i] = tmp;
		i++;
	}

	cudaSetDevice(this->parent_device_id);
	cuLaunchKernel(this->kernel, grid.x, grid.y, grid.y, // grid dim
			block.x, block.y, block.z,                   // block dim
			0, this->streams[stream],                   // shared mem and stream
			args_pointer, 0);                            // arguments
	checkCudaError("Run kernel");
	cudaStreamSynchronize(this->streams[stream]);
	checkCudaError("Synchronize");

}

unsigned int program::get_streams() {
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

