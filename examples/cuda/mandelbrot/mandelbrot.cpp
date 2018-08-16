// Copyright (c)       2017 Madhavan Seshadri
// Copyright (c)       2018 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>

#include <hpxcl/cuda.hpp>

//for writing image
#include "examples/opencl/mandelbrot/pngwriter.cpp"

using std::atoi;
using namespace hpx::cuda;

//###########################################################################
//Main
//###########################################################################

int main(int argc, char* argv[]) {

	//Vector for all futures for the data management
	std::vector<hpx::lcos::future<void>> data_futures;

	//Reading a list of available devices in hpx locality
	std::vector<device> devices = get_all_devices(2, 0).get();

	if (devices.size() < 1) {
		hpx::cerr << "No CUDA devices found! Terminating...." << hpx::endl;
		return hpx::finalize();
	}

	if (argc != 5) {
		std::cout << "Usage: " << argv[0] << "width height" << std::endl;
		exit(1);
	}

	int width = atoi(argv[1]);
	int height = atoi(argv[2]);
	int iterations = atoi(argv[3]);
	int numDevices = atoi(argv[4]);

	//char* image[iterations];
	//char* mainImage[iterations];

    char* image;
    char* mainImage;

    std::vector<hpx::lcos::future<void>> writeImages;

	for (size_t i = 0; i < iterations; i++) {

		int currentWidth = width * (i + 1);
		int currentHeight = height * (i + 1);
		const int bytes = sizeof(char) * currentWidth * currentHeight * 3;

		//Malloc Host
		//char *image;
		cudaMallocHost((void**) &image, bytes);
		checkCudaError("Malloc image");
        //memset(image,0,bytes);
		char* mainImage = (char*)malloc(bytes);
		//char *mainImage;
		//cudaMallocHost((void**) &mainImage, bytes);
		//checkCudaError("Malloc mainImage");
        //memset(mainImage,0,bytes);

		std::vector<hpx::cuda::buffer> args;
		//Generate the grid and block dim
		hpx::cuda::server::program::Dim3 grid;
		hpx::cuda::server::program::Dim3 block;

		block.x = 16;
		block.y = 16;
		block.z = 1;

		grid.x = currentWidth / block.x;
		grid.y = 1 + std::ceil(currentHeight / (block.y * numDevices));
		grid.z = 1;

		std::vector<hpx::future<program>> progBuildVector;
		std::vector<program> progVector;

		//creating vector of futures
		std::vector<hpx::future<void>> kernelFutures;

		for (int j = 0; j < numDevices; j++) {
			progBuildVector.push_back(
					devices[j].create_program_with_file("kernel.cu"));

		}

		hpx::wait_all(progBuildVector);


		std::vector<hpx::lcos::future<void>> progCompileVector;

		for (int j = 0; j < numDevices; j++) {
			progVector.push_back(progBuildVector[j].get());

			//Compile with the kernel
			std::vector < std::string > flags;
			std::string mode = "--gpu-architecture=compute_";
			mode.append(
					std::to_string(
							devices[j].get_device_architecture_major().get()));
			mode.append(
					std::to_string(
							devices[j].get_device_architecture_minor().get()));

			flags.push_back(mode);

			progCompileVector.push_back(progVector[j].build(flags, "kernel"));

		}

		std::vector<hpx::lcos::future<buffer>> bufferFutures;

		//creating buffers
		for (int j = 0; j < numDevices; j++) {
			//Image buffer
			bufferFutures.push_back(devices[j].create_buffer(bytes));
			//Width buffer
			bufferFutures.push_back(devices[j].create_buffer(sizeof(int)));
			//Height buffer
			bufferFutures.push_back(devices[j].create_buffer(sizeof(int)));
			// yStart buffer
			bufferFutures.push_back(devices[j].create_buffer(sizeof(int)));

		}

		hpx::wait_all(bufferFutures);

		//Image Buffer Vector
		std::vector<buffer> imageBufferVector;
		std::vector<buffer> widthBufferVector;
		std::vector<buffer> heightBufferVector;
		std::vector<buffer> yStartBufferVector;

		//creating buffers
		for (int j = 0; j < numDevices; j++) {
			//calculate the start position
			int yStart = j * currentHeight / numDevices;

			imageBufferVector.push_back(bufferFutures[j * 4].get());
			data_futures.push_back(
					imageBufferVector[j].enqueue_write(0, bytes, image));
			widthBufferVector.push_back(bufferFutures[(j * 4) + 1].get());
			data_futures.push_back(
					widthBufferVector[j].enqueue_write(0, sizeof(int), &currentWidth));
			heightBufferVector.push_back(bufferFutures[(j * 4) + 2].get());
			data_futures.push_back(
					heightBufferVector[j].enqueue_write(0, sizeof(int),
							&currentHeight));
			yStartBufferVector.push_back(bufferFutures[(j * 4) + 3].get());
			data_futures.push_back(
					yStartBufferVector[j].enqueue_write(0, sizeof(int),
							&yStart));

		}

		//Synchronize copy to buffer
		hpx::wait_all(data_futures);

		//wait for program to build on all devices
		hpx::wait_all(progCompileVector);

		for (int j = 0; j < numDevices; j++) {

			args.push_back(imageBufferVector[j]);
			args.push_back(widthBufferVector[j]);
			args.push_back(heightBufferVector[j]);
			args.push_back(yStartBufferVector[j]);



			//run the program on the device
#ifdef HPXCL_CUDA_WITH_STREAMS
			kernelFutures.push_back(progVector[j].run(args, "kernel", grid, block, args));
#else
			kernelFutures.push_back(
					progVector[j].run(args, "kernel", grid, block));
#endif
			//for multiple runs
			args.clear();

		}

		//wait for all the kernel futures to return
		hpx::wait_all(kernelFutures);

		//write images to file
		std::shared_ptr<std::vector<char>> img_data;

		//Stich multiple images
		for (int j = 0; j < numDevices; j++) {
			image = imageBufferVector.at(j).enqueue_read_sync<char>(0,
					bytes / numDevices);
			std::copy(image,
					image + currentWidth * (currentHeight / numDevices) * 3
							- 1,
					mainImage +currentWidth * (currentHeight / numDevices) * 3
									* j);
		}
		img_data = std::make_shared < std::vector<char>
				> (mainImage, mainImage + bytes );

		std::string str = "Mandel_brot_imp_";
		str.append(std::to_string(currentWidth));
		str.append("_");
		str.append(std::to_string(currentHeight));
		str.append(".png");
	    writeImages.push_back(hpx::async(save_png, img_data, currentWidth, currentHeight,"Mandelbrot.png"));
	    save_png(img_data, currentWidth, currentHeight, "Mandelbrot_img.png");
    }

    hpx::wait_all(writeImages);

	for (int j = 0; j < numDevices; j++) {
		//Free Memory
		cudaFreeHost(image);
		checkCudaError("Free image");
		cudaFreeHost(mainImage);
		checkCudaError("Free mainImage");
	}

	return EXIT_SUCCESS;
}
