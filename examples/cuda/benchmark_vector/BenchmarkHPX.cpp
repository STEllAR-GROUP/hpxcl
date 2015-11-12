// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "utils.hpp"

#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>

#include "examples/opencl/benchmark_vector/timer.hpp"

#include "hpx_cuda.hpp"

#include "config.hpp"

using namespace hpx::cuda;

//###########################################################################
//Main
//###########################################################################

int main(int argc, char*argv[]) {

	if (argc != 2) {
		std::cout << "Usage: " << argv[0] << " #elements" << std::endl;
		exit(1);
	}

	size_t count = atoi(argv[1]);

	double timeData = 0;

	//Vector for all futures for the data management
	std::vector<hpx::lcos::future<void>> data_futures;

	// Get list of available Cuda Devices.
	std::vector<device> devices = get_all_devices(2, 0).get();

	// Check whether there are any devices
	if (devices.size() < 1) {
		hpx::cerr << "No CUDA devices found!" << hpx::endl;
		return hpx::finalize();
	}

	timer_start();

	//Pointer
	TYPE* out;
	TYPE* in1;
	TYPE* in2;

	timer_start();
	//Malloc Host
	cudaMallocHost((void**) &out, count * sizeof(TYPE));
	cudaMallocHost((void**) &in1, count * sizeof(TYPE));
	cudaMallocHost((void**) &in2, count * sizeof(TYPE));

	//Initialize the data
	fillRandomVector(in1, count);
	fillRandomVector(in2, count);

	// Create a device component from the first device found
	device cudaDevice = devices[0];

	// Create a buffer
	buffer in1Buffer = cudaDevice.create_buffer_sync(count * sizeof(TYPE));
	buffer in2Buffer = cudaDevice.create_buffer_sync(count * sizeof(TYPE));
	buffer outbuffer = cudaDevice.create_buffer_sync(count * sizeof(TYPE));

	// Copy input data to the buffer
	//data_futures.push_back(in1Buffer.enqueue_write(0, SIZE * sizeof(TYPE),
		//			inputData));

	// Create the hello_world device program
	//program prog = cudaDevice.create_program_with_file("./kernel.cu").get();

	// Add compiler flags for compiling the kernel

	//std::vector<std::string> flags;
	//std::string mode = "--gpu-architecture=compute_";
	//mode.append(
	//		std::to_string(cudaDevice.get_device_architecture_major().get()));

	//mode.append(
	//		std::to_string(cudaDevice.get_device_architecture_minor().get()));

	//flags.push_back(mode);

	// Compile the program
	//prog.build(flags);

	// Create the buffer for the result
	//unsigned int* result;
	//cudaMallocHost((void**)&result,sizeof(unsigned int));
	//result[0] = 0;
	//buffer resbuffer = cudaDevice.create_buffer_sync(sizeof(unsigned int));
	//data_futures.push_back(resbuffer.enqueue_write(0,sizeof(unsigned int), result));

	//Create the buffer for the length of the array
	//unsigned int* n;
	//cudaMallocHost((void**)&n,sizeof(unsigned int));
	//result[0] = SIZE;
	//buffer lengthbuffer = cudaDevice.create_buffer_sync(sizeof(unsigned int));
	//data_futures.push_back(lengthbuffer.enqueue_write(0,sizeof(unsigned int), n));

	//Generate the grid and block dim
	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	//Set the values for the grid dimension
	grid.x = 1;
	grid.y = 1;
	grid.z = 1;

	//Set the values for the block dimension
	block.x = 32;
	block.y = 1;
	block.z = 1;

	std::cout << count << " ";
	return EXIT_SUCCESS;
}
