// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include <hpxcl/cuda.hpp>

#define SIZE 100000

using namespace hpx::cuda;

// hpx_main, is the actual main called by hpx
int main(int argc, char* argv[]) {

	//Vector for all futures for the data management
	std::vector<hpx::lcos::future<void>> data_futures;

	// Get list of available Cuda Devices.
	std::vector<device> devices = get_all_devices(2, 0).get();

	// Check whether there are any devices
	if (devices.size() < 1) {
		hpx::cerr << "No CUDA devices found!" << hpx::endl;
		return hpx::finalize();
	}

	// Generate Input data
	unsigned int* inputData;
	cudaMallocHost((void**)&inputData, sizeof(unsigned int)*SIZE);
	checkCudaError("Malloc inputData");

	// Create a device component from the first device found
	device cudaDevice = devices[0];

	for (unsigned int i = 0; i < SIZE; i++)
	inputData[i] = 1;

	// Create a buffer
	buffer outbuffer = cudaDevice.create_buffer(SIZE * sizeof(unsigned int)).get();

	// Copy input data to the buffer
	data_futures.push_back(outbuffer.enqueue_write(0, SIZE * sizeof(unsigned int), inputData));

	// Create the hello_world device program
	program prog = cudaDevice.create_program_with_file("example_kernel.cu").get();

	// Add compiler flags for compiling the kernel

	std::vector<std::string> flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(
	std::to_string(cudaDevice.get_device_architecture_major().get()));

	mode.append(
	std::to_string(cudaDevice.get_device_architecture_minor().get()));

	flags.push_back(mode);

	// Compile the program
	prog.build_sync(flags,"sum");

	// Create the buffer for the result
	unsigned int* result;
	cudaMallocHost((void**)&result,sizeof(unsigned int));
	checkCudaError("Malloc result");
	result[0] = 0;
	buffer resbuffer = cudaDevice.create_buffer(sizeof(unsigned int)).get();
	data_futures.push_back(resbuffer.enqueue_write(0,sizeof(unsigned int), result));

	//Create the buffer for the length of the array
	unsigned int* n;
	cudaMallocHost((void**)&n,sizeof(unsigned int));
	checkCudaError("Malloc size n");
	result[0] = SIZE;
	buffer lengthbuffer = cudaDevice.create_buffer(sizeof(unsigned int)).get();
	data_futures.push_back(lengthbuffer.enqueue_write(0,sizeof(unsigned int), n));

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

	//Set the parameter for the kernel, have to be the same order as in the definition
	std::vector<hpx::cuda::buffer>args;
	args.push_back(outbuffer);
	args.push_back(resbuffer);
	args.push_back(lengthbuffer);

	hpx::wait_all(data_futures);

	//Run the kernel at the default stream
	auto kernel_future = prog.run(args,"sum",grid,block);

	hpx::wait_all(kernel_future);

	//Copy the result back
	unsigned int* res = resbuffer.enqueue_read_sync<unsigned int>(0,sizeof(unsigned int));

	hpx::cout << "Result is " << res[0] << " and is ";

	//Check if result is correct

	if (res[0] != SIZE)
		hpx::cout << "wrong" << hpx::endl;
	else
		hpx::cout << "correct" << hpx::endl;

	cudaFreeHost(n);
	checkCudaError("Free n");
	cudaFreeHost(inputData);
	checkCudaError("Free inputData");
	cudaFreeHost(result);
	checkCudaError("Free result");

	return EXIT_SUCCESS;
}

