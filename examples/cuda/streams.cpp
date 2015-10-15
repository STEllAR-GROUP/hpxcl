// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/wait_each.hpp>

#include "../../cuda.hpp"

#include <unistd.h>

#define SIZE 10000
#define PARTIONS 2

using namespace hpx::cuda;

static const char kernel_src[] =
		        "                                                                                                        "
				"extern \"C\"  __global__ void sum(unsigned int* n, unsigned  int* count,unsigned int* array){ 	       \n"
				" for (int i = blockDim.x * blockIdx.x + threadIdx.x;					                               \n"
				"         i < n[0];														                               \n"
				"         i += gridDim.x * blockDim.x)									                               \n"
				"    {													                                               \n"
				"        atomicAdd(&(count[0]), array[i]);							                                   \n"
				"    }	 									                                                           \n"
				"}                                             							                               \n";

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

	// Create a device component from the first device found
	device cudaDevice = devices[0];

	// Create the hello_world device program
	program prog = cudaDevice.create_program_with_source(kernel_src).get();

	// Add compiler flags for compiling the kernel

	std::vector<std::string> flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(
	std::to_string(cudaDevice.get_device_architecture_major().get()));

	mode.append(
	std::to_string(cudaDevice.get_device_architecture_minor().get()));

	flags.push_back(mode);

	// Compile the program
	prog.build(flags);

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

	// Generate Input data
	unsigned int* inputData;
	cudaMallocHost((void**)&inputData, sizeof(unsigned int)*SIZE);

	for (unsigned int i = 0; i < SIZE; i++)
	inputData[i] = 1;

	//Create buffer for the result
	unsigned int result = 0;
	std::vector<buffer> resultBuffer;
	std::vector<hpx::future<void>> syncFutures;
	for (unsigned int i = 0; i < PARTIONS; i++ )
	{
		resultBuffer.push_back(cudaDevice.create_buffer_sync(sizeof(unsigned int)));
		syncFutures.push_back(resultBuffer[i].enqueue_write(0,sizeof(unsigned int), &result));
	}

	//Create a buffer for the sliced size of the data
	unsigned int slicedSize = SIZE / PARTIONS;
	buffer sizeBuffer = cudaDevice.create_buffer_sync(sizeof(unsigned int));
	syncFutures.push_back(sizeBuffer.enqueue_write(0,sizeof(unsigned int), &slicedSize));

	//Sync all meta data for the launch of the kernels
	hpx::wait_all(syncFutures);

	hpx::cout << "Running: " << SIZE << " elements sliced on " << PARTIONS << " cudaStreams with " << slicedSize << " elements" << hpx::endl;

	//Set the parameter for the kernel, have to be the same order as in the definition
	std::vector<hpx::cuda::buffer>args;
	args.push_back(sizeBuffer);

	//Distribute the sliced data on the different kernel streams
	std::vector<buffer>slicedResults;
	std::vector<hpx::lcos::future<unsigned int>> results;

	for (unsigned int i=0; i < PARTIONS; i++ )
	{
		unsigned int stream_id = prog.create_stream().get();
		slicedResults.push_back(cudaDevice.create_buffer_sync(slicedSize *sizeof(unsigned int)));

		auto f = slicedResults[i].enqueue_write(0,slicedSize *sizeof(unsigned int),&inputData[slicedSize]);

		args.push_back(resultBuffer[i]);
        args.push_back(slicedResults[i]);

		prog.run(args,"sum",grid,block);

		//results.push_back(f2.then(resultBuffer[i].enqueue_read(0,sizeof(unsigned int))));

		args.pop_back();
		args.pop_back();


	}


	hpx::wait_all(results);

	return EXIT_SUCCESS;
}

