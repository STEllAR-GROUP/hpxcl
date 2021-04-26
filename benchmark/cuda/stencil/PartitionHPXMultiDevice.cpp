// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include "examples/opencl/benchmark_vector/timer.hpp"

#include <hpxcl/cuda.hpp>

#include "config.hpp"
#include "utils.hpp"

using namespace hpx::cuda;

//###########################################################################
//Kernels
//###########################################################################

static const char kernel_src[] =

"extern \"C\" __global__ void kernel(float* in) { 		 				\n"
		"																\n"
		"	size_t i = threadIdx.x + blockIdx.x * blockDim.x;			\n"
		"	float x = (float) i;										\n"
		"	float s = sinf(x);											\n"
		"	float c = cosf(x);											\n"
		"	in[i] = in[i] + sqrtf(s * s + c * c);						\n"
		"																\n"
		"}																\n";

//###########################################################################
//Main
//###########################################################################

int main(int argc, char*argv[]) {

	// Get list of available Cuda Devices.
	std::vector<device> devices = get_all_devices(2, 0).get();

	// Check whether there are any devices
	if (devices.size() < 1) {
		hpx::cerr << "No CUDA devices found!" << hpx::endl;
		return hpx::finalize();
	}

	const int blockSize = 256;


	if (argc != 3) {
		std::cout << "Usage: " << argv[0] << " n -> 2^n*1024*" << blockSize
				<< " elements and g -> amount of GPUs" << std::endl;
		exit(1);
	}

	double time = 0;
	size_t count = atoi(argv[1]);
	int deviceCount = atoi(argv[2]);

	if( deviceCount > devices.size())
    {
        std::cout << "Error: You are using " << deviceCount << " and only "
            << devices.size() << "are available" << std::endl;
        exit(1);
    }


	const int n = pow(2,count) * 1024 * blockSize;
	const int streamSize = n / deviceCount;
	const int streamBytes = streamSize * sizeof(TYPE);
	const int bytes = n * sizeof(TYPE);

	std::cout << n << " ";

	timer_start();

	//Malloc Host
	TYPE* in;
	cudaMallocHost((void**) &in, bytes);
	checkCudaError("Malloc in");
	for (size_t i = 0; i < n ; i++)
	        in[i] = 0.;

	std::vector<hpx::future<void>> dependencies;

	// Create the hello_world device program
	std::vector<hpx::future<hpx::cuda::program>> fprogram;

	for (size_t i = 0; i < deviceCount; i++)
	fprogram.push_back(devices[i].create_program_with_source(kernel_src));

	// Add compiler flags for compiling the kernel

	std::vector<std::string> flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(
			std::to_string(devices[0].get_device_architecture_major().get()));
	mode.append(
			std::to_string(devices[0].get_device_architecture_minor().get()));

	flags.push_back(mode);

	hpx::wait_all(fprogram);

	std::vector<hpx::cuda::program> programs;
	for (size_t i = 0; i < deviceCount; i++)
	{
		programs.push_back(fprogram[i].get());
		dependencies.push_back(programs[i].build(flags, "kernel"));

	}

    std::vector<hpx::future<hpx::cuda::buffer>> fbuffer;
    for (size_t i = 0; i < deviceCount; i++)
    {                                                                            
       fbuffer.push_back(devices[i].create_buffer(streamBytes));
    }                                                                            
        
    hpx::wait_all(fbuffer);

	std::vector<buffer> bufferIn;
	for (size_t i = 0; i < deviceCount; i++)
	{
	   bufferIn.push_back(fbuffer[i].get());
	}

	for (size_t i = 0; i < deviceCount; i++)
	{

	    dependencies.push_back(bufferIn[i].enqueue_write(i*streamSize,streamBytes,in));
    }
    
	std::vector<hpx::cuda::buffer> args;
	//Generate the grid and block dim
	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	//Set the values for the grid dimension
	grid.x = streamSize / blockSize;
	grid.y = 1;
	grid.z = 1;

	//Set the values for the block dimension
	block.x = blockSize;
	block.y = 1;
	block.z = 1;

	hpx::wait_all(dependencies);

	std::vector<hpx::future<void>> kernelFutures;
	for (size_t i = 0; i < deviceCount; i++)
	{
		args.push_back(bufferIn[i]);
		#ifdef HPXCL_CUDA_WITH_STREAMS
		kernelFutures.push_back(programs[i].run(args, "kernel", grid, block,args));
		#else
		kernelFutures.push_back(programs[i].run(args, "kernel", grid, block));
		#endif
		args.clear();
	}

	hpx::wait_all(kernelFutures);

	time += timer_stop();

	bool check;
	for (size_t i = 0; i < deviceCount; i++)
	{
		TYPE* res = bufferIn[i].enqueue_read_sync<TYPE>(0,streamBytes);
		check = checkKernel(res,streamSize);
		if(check==false) break;

	}

	timer_start();

	//Clean
	cudaFreeHost(in);
	checkCudaError("Free in");

	std:: cout << check << " " << time + timer_stop() << std::endl;

	return EXIT_SUCCESS;
}

