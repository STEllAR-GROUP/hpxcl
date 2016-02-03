// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)



#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>

#include "examples/opencl/benchmark_vector/timer.hpp"

#include "hpx_cuda.hpp"

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

	size_t* count;
	cudaMallocHost((void**)&count,sizeof(size_t));
	count[0]= atoi(argv[1]);

	std::cout << count[0] << " ";

	//Vector for all futures for the data management
	std::vector<hpx::lcos::future<void>> data_futures;

	// Get list of available Cuda Devices.
	std::vector<device> devices = get_all_devices(1, 0).get();

	// Check whether there are any devices
	if (devices.size() < 1) {
		hpx::cerr << "No CUDA devices found!" << hpx::endl;
		return hpx::finalize();
	}

	timer_start();

	//Pointer
	TYPE* out;
	TYPE* in;
	TYPE* s;

	double data = 0.;
	timer_start();
	//Malloc Host
	cudaMallocHost((void**) &out, count[0] * sizeof(TYPE));
	cudaMallocHost((void**) &in, count[0] * sizeof(TYPE));
	cudaMallocHost((void**) &s, 3 * sizeof(TYPE));

	data += timer_stop();

	//Initialize the data
	fillRandomVector(in, count[0]);
	s[0] = 0.5;
	s[1] = 1.;
	s[2] = 0.5;

	// Create a device component from the first device found
	device cudaDevice = devices[0];

	// Create a buffer
	timer_start();
	buffer inBuffer = cudaDevice.create_buffer_sync(count[0] * sizeof(TYPE));
	buffer sBuffer = cudaDevice.create_buffer_sync(3 * sizeof(TYPE));
	buffer outBuffer = cudaDevice.create_buffer_sync(count[0] * sizeof(TYPE));
	buffer lengthbuffer = cudaDevice.create_buffer_sync(sizeof(size_t));

	// Copy input data to the buffer
	data_futures.push_back(inBuffer.enqueue_write(0, count[0] * sizeof(TYPE),
					in));
	data_futures.push_back(sBuffer.enqueue_write(0, 3 * sizeof(TYPE),
					s));
	data_futures.push_back(outBuffer.enqueue_write(0, count[0] * sizeof(TYPE),
					in));
	data_futures.push_back(lengthbuffer.enqueue_write(0,sizeof(size_t), count));

	hpx::wait_all(data_futures);

	data += timer_stop();

	std::cout << data << " ";

	timer_start();
	// Create the hello_world device program
	program prog = cudaDevice.create_program_with_file("kernel.cu").get();

	//Add compiler flags for compiling the kernel
	std::vector<std::string> flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(
			std::to_string(cudaDevice.get_device_architecture_major().get()));

	mode.append(
			std::to_string(cudaDevice.get_device_architecture_minor().get()));

	flags.push_back(mode);
	flags.push_back("-use_fast_math");

	// Compile the program
	prog.build_sync(flags,"stencil");



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
	std::vector<hpx::cuda::buffer>args;
	args.push_back(lengthbuffer);
	args.push_back(inBuffer);
	args.push_back(outBuffer);
	args.push_back(sBuffer);

	auto kernel_future = prog.run(args,"stencil",grid,block);

	hpx::wait_all(kernel_future);

	std::cout << timer_stop() << " ";

	timer_start();

	TYPE* res = outBuffer.enqueue_read_sync<TYPE>(0,sizeof(TYPE));

	std::cout << timer_stop() << " ";

	//Check the result
	std::cout << checkStencil(in,res,s, count[0]) << " ";

	//Cleanup
	timer_start();
	cudaFreeHost(in);
	cudaFreeHost(s);
	cudaFreeHost(out);
	cudaFreeHost(count);

	std::cout << timer_stop() << std::endl;

	return EXIT_SUCCESS;
}
