// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/hpx_main.hpp>

#include <hpxcl/cuda.hpp>

#include <hpxcl/examples/opencl/mandelbrot/pngwriter.hpp>

using namespace hpx::cuda;

//###########################################################################
//Main
//###########################################################################

int main(int argc, char* argv[]){

	//Reading a list of available devices in hpx locality
	//Returns a future
	std::std::vector<device> devices = get_all_devices(2, 0).get();

	if(devices.size() < 1) {
		hpx::cerr << "No CUDA devices found! Terminating...." << hpx::endl;
		return hpx::finalize();
	}
	
	if (argc != 3) {
		std::cout << "Usage: " << argv[0] << " #iterations width height"<< std::endl;
		exit(1);
	}

	const int bytes = sizeof(char) * width * height * 3;

	//Malloc Host
	char *image;
	cudaMallocHost((void**) &image, bytes);
	memset(in, 0, bytes);

	//Creating a new device from the list of devices
	device cudaDevice = devices[0];

	//Create a Mandelbrot device program
	program prog = cudaDevice.create_program_with_file("kernel.cu");

	//Compile with the kernal
	std::vector<std::string> flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(
			std::to_string(cudaDevice.get_device_architecture_major().get()));
	mode.append(
			std::to_string(cudaDevice.get_device_architecture_minor().get()));

	flags.push_back(mode);

	auto f = prog.build(flags, "kernel");

	std::vector<buffer> bufferIn;
	
	bufferIn.push_back(cudaDevice.create_buffer(bytes));

	char *mainImage = (char*) malloc(bytes);

	std::vector<hpx::cuda::buffer> args;
	//Generate the grid and block dim
	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	block.x = 16;
	block.y = 16;
	block.z = 1;

	grid.x = width/block.x;
	grid.y = height/block.y;
	grid.z = 1;

	//creating vector of futures
	std::std::vector<hpx::future<void>> kernelFutures;
	hpx::wait_all(f);

	args.push_back(bufferIn[0]);
	kernelFutures.push_back(prog.run(args, "kernel", grid, block, args));
	args.clear();

	hpx::wait_all(kernelFutures);

	//Free Memory
	cudaFree(image);
	free(mainImage);

	return EXIT_SUCCESS;
}