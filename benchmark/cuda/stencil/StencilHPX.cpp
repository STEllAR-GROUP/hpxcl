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
//Main
//###########################################################################

int main(int argc, char*argv[]) {

	if (argc != 2) {
		std::cout << "Usage: " << argv[0] << " #elements" << std::endl;
		exit(1);
	}

	double data = 0.;

	timer_start();
	size_t* count;
	cudaMallocHost((void**)&count,sizeof(size_t));
	checkCudaError("Malloc count");
	count[0]= atoi(argv[1]);

	std::cout << count[0] << " ";

	//Vector for all futures for the data management
	std::vector<hpx::lcos::future<void>> data_futures;

	// Get list of available Cuda Devices.
	std::vector<device> devices = get_all_devices(1, 0).get();
	data += timer_stop();
	// Check whether there are any devices
	if (devices.size() < 1) {
		hpx::cerr << "No CUDA devices found!" << hpx::endl;
		return hpx::finalize();
	}

	//Pointer
	TYPE* out;
	TYPE* in;
	TYPE* s;

	//Malloc Host
	cudaMallocHost((void**) &out, count[0] * sizeof(TYPE));
	checkCudaError("Mallloc out");
	cudaMallocHost((void**) &in, count[0] * sizeof(TYPE));
	checkCudaError("Malloc in");
	cudaMallocHost((void**) &s, 3 * sizeof(TYPE));
	checkCudaError("Malloc s");

	//Initialize the data
	fillRandomVector(in, count[0]);
	s[0] = 0.5;
	s[1] = 1.;
	s[2] = 0.5;

	// Create a device component from the first device found
	device cudaDevice = devices[0];

	// Create the hello_world device program
	hpx::lcos::future < hpx::cuda::program > futureProg = cudaDevice.create_program_with_file(
			"stencil_kernel.cu");

	//Add compiler flags for compiling the kernel
	std::vector < std::string > flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(
			std::to_string(cudaDevice.get_device_architecture_major().get()));

	mode.append(
			std::to_string(cudaDevice.get_device_architecture_minor().get()));

	flags.push_back(mode);
	flags.push_back("-use_fast_math");

	// Compile the program
	hpx::cuda::program prog = futureProg.get();
	prog.build_sync(flags, "stencil");

	// Create a buffer
	std::vector<hpx::lcos::future<buffer>> futureBuffers;
	futureBuffers.push_back(cudaDevice.create_buffer(count[0] * sizeof(TYPE)));
	futureBuffers.push_back(cudaDevice.create_buffer(3 * sizeof(TYPE)));
	futureBuffers.push_back(cudaDevice.create_buffer(count[0] * sizeof(TYPE)));
	futureBuffers.push_back(cudaDevice.create_buffer(sizeof(size_t)));

	hpx::wait_all(futureBuffers);

	buffer inBuffer = futureBuffers[0].get();
	buffer sBuffer = futureBuffers[1].get();
	buffer outBuffer = futureBuffers[2].get();
	buffer lengthbuffer = futureBuffers[3].get();

	// Copy input data to the buffer
	data_futures.push_back(
			inBuffer.enqueue_write(0, count[0] * sizeof(TYPE), in));
	data_futures.push_back(sBuffer.enqueue_write(0, 3 * sizeof(TYPE), s));
	data_futures.push_back(
			outBuffer.enqueue_write(0, count[0] * sizeof(TYPE), in));
	data_futures.push_back(
			lengthbuffer.enqueue_write(0, sizeof(size_t), count));

	//Generate the grid and block dim
	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	//Set the values for the grid dimension
	grid.x = 1;
	grid.y = 0;
	grid.z = 0;

	//Set the values for the block dimension
	block.x = 32;
	block.y = 0;
	block.z = 0;

	//Launch kernels
	std::vector<hpx::cuda::buffer> args;
	args.push_back(lengthbuffer);
	args.push_back(inBuffer);
	args.push_back(outBuffer);
	args.push_back(sBuffer);

	data_futures.push_back(prog.run(args, "stencil", grid, block));

    hpx::wait_all(data_futures); 

	TYPE* res = outBuffer.enqueue_read_sync<TYPE>(0, sizeof(TYPE));

	data += timer_stop();

	//Check the result
	std::cout << checkStencil(in, res, s, count[0]) << " ";

	timer_start();

	//Cleanup
	cudaFreeHost(in);
	checkCudaError("Free in");
	cudaFreeHost(s);
	checkCudaError("Free s");
	cudaFreeHost(out);
	checkCudaError("Free out");
	cudaFreeHost(count);
	checkCudaError("Free count");

	std::cout << data + timer_stop() << std::endl;

	return EXIT_SUCCESS;
}
