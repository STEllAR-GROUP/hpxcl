// Copyright (c)       2021 Patrick Diehl
//                     2021 Pedro Barbosa 
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include <hpxcl/cuda.hpp>

#include <chrono>

using namespace hpx::cuda;

#define SIZE 64

int main(int argc, char* argv[]) {

	auto start = std::chrono::steady_clock::now();

	std::vector<hpx::lcos::future<void>> data_futures;

	std::vector<device> devices = get_all_devices(2, 0).get();

	if (devices.size() < 1) {
		hpx::cerr << "No CUDA devices found!" << hpx::endl;
		return hpx::finalize();
	}


	int* input;
	cudaMallocHost((void**)&input, sizeof(int) * SIZE);
	checkCudaError("Malloc inputData");

	for(int i = 0; i < SIZE; i++){
		input[i] = i;
		std::cout << input[i] << ", ";
	}
	std::cout << std::endl;


	device cudaDevice = devices[0];

	buffer inbuffer = cudaDevice.create_buffer(sizeof(int) * SIZE).get();

	data_futures.push_back(inbuffer.enqueue_write(0, sizeof(int) * SIZE, input));

	program prog = cudaDevice.create_program_with_file("example_shared_kernel.cu").get();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Add compiler flags for compiling the kernel
	std::vector<std::string> flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(std::to_string(cudaDevice.get_device_architecture_major().get()));
	mode.append(std::to_string(cudaDevice.get_device_architecture_minor().get()));
	flags.push_back(mode);

	// Compile the program
	prog.build_sync(flags, "dynamicReverse");
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Generate the grid and block dim
	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	// Set the values for the grid dimension
	grid.x = 1;
	grid.y = 1;
	grid.z = 1;

	// Set the values for the block dimension
	block.x = SIZE;
	block.y = 1;
	block.z = 1;



	int* n;
	cudaMallocHost((void**)&n, sizeof(int));
	n[0] = SIZE;

	buffer sizebuffer = cudaDevice.create_buffer(sizeof(int)).get();
	data_futures.push_back(sizebuffer.enqueue_write(0, sizeof(int), n));



	// Set the parameter for the kernel, have to be the same order as in the definition
	std::vector<hpx::cuda::buffer> args;
	args.push_back(inbuffer);
	args.push_back(sizebuffer);

	hpx::wait_all(data_futures);

	//Run the kernel at the default stream
	auto kernel_future = prog.run(args, "dynamicReverse", grid, block, SIZE*sizeof(int));

	hpx::wait_all(kernel_future);

	
	//Copy the result back
	int* res = inbuffer.enqueue_read_sync<int>(0, SIZE*sizeof(int));

	for (int i = 0; i < SIZE; i++){
		std::cout << res[i] << ", ";
	}
	std::cout << std::endl;



	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

	return EXIT_SUCCESS;
}


