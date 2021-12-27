// Copyright (c)       2021 Pedro Barbosa
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include <hpxcl/cuda.hpp>

#include <chrono>

using namespace hpx::cuda;


int main (int argc, char* argv[]) {
	std::cout << "Start" << std::endl;
	auto start = std::chrono::steady_clock::now();

	//Vector for all futures for the data management
	std::vector<hpx::lcos::future<void>> data_futures;

	// Get list of available Cuda Devices
	std::vector<device> devices = get_all_devices(2, 0).get();

	// Check whether there are any devices
	if (devices.size() < 1) {
		hpx::cerr << "No CUDA devices found!" << hpx::endl;
		return hpx::finalize();
	}

	// Create a device component from the first device found
	device cudaDevice = devices[0];

	// Generate Input data for the device
	int* array;
	cudaMallocHost((void**)&array, sizeof(int) * 8);
	checkCudaError("Malloc array");

	for(int i = 0; i < 8; i++){
		array[i] = 1;
	}

	// Create a buffer
	buffer array_buffer = cudaDevice.create_buffer(sizeof(int) * 8).get();

	// Copy input data to the buffer
	data_futures.push_back(array_buffer.enqueue_write(0, sizeof(int) * 8, array));



	// Create the writeTest_kernel device program
	program prog = cudaDevice.create_program_with_file("writeTest_kernel.cu").get();

	// Add compiler flags for compiling the kernel on the device
	std::vector<std::string> flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(std::to_string(cudaDevice.get_device_architecture_major().get()));
	mode.append(std::to_string(cudaDevice.get_device_architecture_minor().get()));
	flags.push_back(mode);

	// Compile the program
	prog.build_sync(flags, "writeTest");

	// Generate the grid and block dim
	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	// Set the values for the grid dimension
	grid.x = 1;
	grid.y = 1;
	grid.z = 1;

	// Set the values for the block dimension
	block.x = 1;
	block.y = 1;
	block.z = 1;

	// Set the parameter for the kernel, have to be the same order as in the
  	// definition
	std::vector<hpx::cuda::buffer> args;
	args.push_back(array_buffer);


	// Generate new Input data to test the new write function
	int* small_array;
	cudaMallocHost((void**)&small_array, sizeof(int) * 4);
	checkCudaError("Malloc small_array");


	small_array[0] = 8;
	small_array[1] = 8;
	small_array[2] = 16;
	small_array[3] = 16;


	// Copy input data to the buffer with the destination and source offset
	data_futures.push_back(array_buffer.enqueue_write_parcel(sizeof(int) * 4, sizeof(int) * 2, small_array, sizeof(int) * 1));


	hpx::wait_all(data_futures);


	// Run the kernel at the default stream on the device
	auto kernel_future = prog.run(args, "writeTest", grid, block, 0);
	kernel_future.get();

  	// Copy the result back
	int* res;
	cudaMallocHost((void**)&res, sizeof(int) * 8);
	res = array_buffer.enqueue_read_sync<int>(0, sizeof(int) * 8);

	for(int i = 0; i < 8; i++){
		std::cout << res[i] << std::endl;
	}


	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	return EXIT_SUCCESS;
}
