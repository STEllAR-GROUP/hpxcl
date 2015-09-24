// Copyright (c)    2013 Damond Howard
//                  2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include <hpx/hpx.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda/server/program.hpp"
#include "cuda/kernel.hpp"

#include <sstream>
#include <string>

namespace hpx {
namespace cuda {
namespace server {

program::program() {

}

program::program(int parent_device_id) {
	this->parent_device_id = parent_device_id;
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	this->streams.push_back(stream);
}

program::program(hpx::naming::id_type device_id, std::string code) {
}

program::program(hpx::naming::id_type device_id,
		hpx::serialization::serialize_buffer<char> binary) {
}

program::~program() {

	nvrtcDestroyProgram (&prog);
	checkCudaError("Destroy Program");

	for (auto stream : streams)
	{
		cudaStreamDestroy(stream);
		checkCudaError("Destroy stream");
	}
}

void program::set_source(std::string source) {
	this->kernel_source = source;
}
hpx::cuda::kernel program::create_kernel(std::string module_name,
		std::string kernel_name) {
	typedef hpx::cuda::server::kernel kernel_type;

	hpx::cuda::kernel cu_kernel(
			hpx::components::new_ < kernel_type
					> (hpx::find_here(), reinterpret_cast<intptr_t>(&this->kernel)));
	//cu_kernel.load_module_sync(module_name);
	//cu_kernel.load_kernel_sync(kernel_name);
	return cu_kernel;
}

//ToDo: Add debug flag
void program::build(std::vector<std::string> compilerFlags,
		unsigned int debug) {

	boost::uuids::uuid uuid = boost::uuids::random_generator()();
	std::string filename = to_string(uuid);
	filename.append(".cu");

	if (debug == 1) {
		compilerFlags.push_back("-G");
		compilerFlags.push_back("-lineinfo");
	}

	nvrtcCreateProgram(&(this->prog), this->kernel_source.c_str(),
			filename.c_str(), 0, NULL, NULL);
	checkCudaError("Create Program");
	const char * opts[compilerFlags.size()];
	unsigned int i = 0;
	for (auto opt : compilerFlags) {
		opts[i] = compilerFlags[i].c_str();
		i++;
	}


	nvrtcResult compileResult = nvrtcCompileProgram(this->prog,
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
	std::cout << ptxSize << std::endl;
	checkCudaError("Get ptx size");

	char *ptx = new char[ptxSize];
	nvrtcGetPTX(prog, ptx);
	checkCudaError("Get ptx of Program");

	//CUdevice cuDevice;
	//CUcontext context;
	CUmodule module;

	//cuDeviceGet(&cuDevice, this->parent_device_id);
	////checkCudaError("Get Device");
	//cuCtxCreate(&context, 0, cuDevice);
	//checkCudaError("Create Context");
	cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
	checkCudaError("Load Module");
	cuModuleGetFunction(&(this->kernel), module, "sum");
	checkCudaError("Get Function");
	std::cout << kernel << std::endl;
	checkCudaError("Get Function");
}

void program::run(std::vector<intptr_t> args, std::string modulename, Dim3 grid, Dim3 block, unsigned int stream) {

	void *args_pointer[args.size()];

	unsigned int i = 0;
	for (auto arg : args) {

		hpx::cuda::server::buffer* buffer_tmp =
				reinterpret_cast<hpx::cuda::server::buffer*>(&arg);
		CUdeviceptr dev_pointer = buffer_tmp->get_raw_pointer();
		args_pointer[i] = (void*)&dev_pointer;
	}
	//cudaSetDevice(0);
	//std::cout << kernel << std::endl;
	cuLaunchKernel(this->kernel, grid.x, grid.y, grid.y, // grid dim
			block.x, block.y, block.z,    // block dim
			0, this->streams[stream],             // shared mem and stream
			NULL, 0);   // arguments
	checkCudaError("Run kernel");
	cudaDeviceSynchronize();
	checkCudaError("Synchronize");
	//std::cout << "execute kernel " << function << std::endl ;
}

}
}
}

