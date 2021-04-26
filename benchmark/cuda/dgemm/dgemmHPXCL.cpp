// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include <hpxcl/cuda.hpp>

#include "examples/opencl/benchmark_vector/timer.hpp"
#include "validation.h"

using namespace hpx::cuda;

//###########################################################################
//Main
//###########################################################################
int main(int argc, char*argv[]) {

	if (argc != 4) {
		std::cout << "Usage: " << argv[0] << " #m #n #k";
		exit(1);
	}

	int m, n, k, i;

	//Initilizing the matrix dimensions
	m = atoi(argv[1]);
	n = atoi(argv[2]);
	k = atoi(argv[3]);

	double time = 0;
	timer_start();

	//Vector for all futures for the data management
	std::vector<hpx::lcos::future<void>> data_futures;

	// Get list of available Cuda Devices.
	std::vector<device> devices = get_all_devices(2, 0).get();

	// Check whether there are any devices
	if (devices.size() < 1) {
		hpx::cerr << "No CUDA devices found!" << hpx::endl;
		return hpx::finalize();
	}

	double *A, *B, *C;

	double alpha, beta;

	//initializing values of alpha and beta
	alpha = 1.0;
	beta = 0.0;

	//Malloc Host
	cudaMallocHost((void**) &A, m * k * sizeof(double));
	checkCudaError("Malloc A");
	cudaMallocHost((void**) &B, n * k * sizeof(double));
	checkCudaError("Malloc B");
	cudaMallocHost((void**) &C, m * n * sizeof(double));
	checkCudaError("Malloc C");

	time += timer_stop();
	//printf (" Intializing matrix data \n\n");
	timer_start();

	for (i = 0; i < (m * k); i++) {
		A[i] = (double) (i + 1);
	}

	for (i = 0; i < (k * n); i++) {
		B[i] = (double) (-i - 1);
	}

	for (i = 0; i < (m * n); i++) {
		C[i] = 0.0;
	}

	//creating vector of futures
	std::vector<hpx::future<void>> kernelFutures;

	std::vector<hpx::cuda::buffer> args;
	//Generate the grid and block dim
	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	block.x = 32;
	block.y = 32;
	block.z = 1;

	grid.x = 1 + std::ceil(m / block.x);
	grid.y = 1 + std::ceil(n / block.y);
	grid.z = 1;

	std::vector<hpx::future<void>> progBuildVector;
	std::vector<program> progVector;
	std::vector<device> deviceVector;

	//Creating the first device found
	device cudaDevice = devices[0];

	//Create a Mandelbrot device program
	hpx::lcos::future<program> fProg = cudaDevice.create_program_with_file(
			"dgemm.cu");

	//Compile with the kernal
	std::vector < std::string > flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(
			std::to_string(cudaDevice.get_device_architecture_major().get()));
	mode.append(
			std::to_string(cudaDevice.get_device_architecture_minor().get()));

	flags.push_back(mode);

	program prog = fProg.get();
	progBuildVector.push_back(prog.build(flags, "dgemm"));
	progVector.push_back(prog);
	deviceVector.push_back(cudaDevice);

	//wait for program to build on all devices
	hpx::wait_all(progBuildVector);

	//creating buffers
	hpx::lcos::future<hpx::cuda::buffer> fABuffer = cudaDevice.create_buffer(
			m * k * sizeof(double));
	hpx::lcos::future<hpx::cuda::buffer> fBBuffer = cudaDevice.create_buffer(
			n * k * sizeof(double));
	hpx::lcos::future<hpx::cuda::buffer> fCBuffer = cudaDevice.create_buffer(
			m * n * sizeof(double));
	hpx::lcos::future<hpx::cuda::buffer> falphaBuffer =
			cudaDevice.create_buffer(sizeof(double));
	hpx::lcos::future<hpx::cuda::buffer> fbetaBuffer = cudaDevice.create_buffer(
			sizeof(double));
	hpx::lcos::future<hpx::cuda::buffer> fmBuffer = cudaDevice.create_buffer(
			sizeof(int));
	hpx::lcos::future<hpx::cuda::buffer> fnBuffer = cudaDevice.create_buffer(
			sizeof(int));
	hpx::lcos::future<hpx::cuda::buffer> fkBuffer = cudaDevice.create_buffer(
			sizeof(int));

	buffer ABuffer = fABuffer.get();
	data_futures.push_back(ABuffer.enqueue_write(0, m * k * sizeof(double), A));
	buffer BBuffer = fBBuffer.get();
	data_futures.push_back(BBuffer.enqueue_write(0, n * k * sizeof(double), B));
	buffer CBuffer = fCBuffer.get();
	data_futures.push_back(CBuffer.enqueue_write(0, m * n * sizeof(double), C));
	buffer mBuffer = fmBuffer.get();
	data_futures.push_back(mBuffer.enqueue_write(0, sizeof(int), &m));
	buffer nBuffer = fnBuffer.get();
	data_futures.push_back(nBuffer.enqueue_write(0, sizeof(int), &n));
	buffer kBuffer = fkBuffer.get();
	data_futures.push_back(kBuffer.enqueue_write(0, sizeof(int), &k));
	buffer alphaBuffer = falphaBuffer.get();
	data_futures.push_back(
			alphaBuffer.enqueue_write(0, sizeof(double), &alpha));
	buffer betaBuffer = fbetaBuffer.get();
	data_futures.push_back(betaBuffer.enqueue_write(0, sizeof(double), &beta));

	//Synchronize copy to buffer
	hpx::wait_all(data_futures);

	args.push_back(ABuffer);
	args.push_back(BBuffer);
	args.push_back(CBuffer);
	args.push_back(mBuffer);
	args.push_back(nBuffer);
	args.push_back(kBuffer);
	args.push_back(alphaBuffer);
	args.push_back(betaBuffer);

	//Synchronize data transfer before new data is written
	hpx::wait_all(args);

	//run the program on the device
#ifdef HPXCL_CUDA_WITH_STREAMS
	kernelFutures.push_back(prog.run(args, "dgemm", grid, block, args));
#else
	kernelFutures.push_back(prog.run(args, "dgemm", grid, block));
#endif

	//wait for all the kernal futures to return
	hpx::wait_all(kernelFutures);


	double* res = CBuffer.enqueue_read_sync<double>(0, n * m * sizeof(double));

	//Printing the end timing result
	time += timer_stop();
	std::cout << time << " ";

	// Validating the result
	std::cout << validateDgemm(A, B, res, alpha, beta, n, m, k) << std::endl;

	//Free Memory
	args.clear();
	cudaFreeHost(A);
	checkCudaError("Free A");
	cudaFreeHost(B);
	checkCudaError("Free B");
	cudaFreeHost(C);
	checkCudaError("Free C");

	return 0;
}
