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

	if (argc != 3) {
		std::cout << "Usage: " << argv[0] << " #m #n";
		exit(1);
	}

	int m, n, i;

	//Initializing the matrix dimensions
	m = atoi(argv[1]);
	n = atoi(argv[2]);

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
	double *A_data;
	int *A_indices, *A_pointers;

	double alpha;

	//Initializing values of alpha and beta
	alpha = 1.0;

	//Malloc Host
	cudaMallocHost((void**) &A, m * n * sizeof(double));
	checkCudaError("svmp malloc A");
	int count = 0;
	//Input can be anything sparse
	for (i = 0; i < (m * n); i++) {
		if ((i % n) == 0) {
			A[i] = (double) (i + 1);
			count++;
		}
	}

	cudaMallocHost((void**) &B, n * 1 * sizeof(double));
	checkCudaError("svmp malloc B");
	cudaMallocHost((void**) &C, m * 1 * sizeof(double));
	checkCudaError("svmp malloc C");
	cudaMallocHost((void**) &A_data, count * sizeof(double));
	checkCudaError("svmp malloc A_data");
	cudaMallocHost((void**) &A_indices, count * sizeof(int));
	checkCudaError("svmp malloc A_pointers");
	cudaMallocHost((void**) &A_pointers, m * sizeof(int));

	for (i = 0; i < (1 * n); i++) {
		B[i] = (double) (-i - 1);
	}

	for (i = 0; i < (m * 1); i++) {
		C[i] = 0.0;
	}

	//Counters for compression
	int data_counter = 0;
	int index_counter = 0;
	int pointer_counter = -1;

	//Compressing Matrix A
	for (i = 0; i < (m * n); i++) {
		if (A[i] != 0) {
			A_data[data_counter++] = A[i];
			if (((int) i / n) != pointer_counter)
				A_pointers[++pointer_counter] = index_counter;
			A_indices[index_counter++] = (i % n);
		}
	}

	//creating vector of futures
	std::vector<hpx::future<void>> kernelFutures;

	std::vector<hpx::cuda::buffer> args;
	//Generate the grid and block dim
	hpx::cuda::server::program::Dim3 grid;
	hpx::cuda::server::program::Dim3 block;

	block.x = 32;
	block.y = 1;
	block.z = 1;

	grid.x = 1 + std::ceil(m / block.x);
	grid.y = 1;
	grid.z = 1;

	std::vector<hpx::future<void>> progBuildVector;
	std::vector<program> progVector;
	std::vector<device> deviceVector;

	//Creating the first device found
	device cudaDevice = devices[0];

	//Create a Mandelbrot device program
	hpx::lcos::future < program > fProg = cudaDevice.create_program_with_file(
			"smvp.cu");

	//Compile with the kernal
	std::vector < std::string > flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(
			std::to_string(cudaDevice.get_device_architecture_major().get()));
	mode.append(
			std::to_string(cudaDevice.get_device_architecture_minor().get()));

	flags.push_back(mode);

	program prog = fProg.get();
	progBuildVector.push_back(prog.build(flags, "smvp"));
	progVector.push_back(prog);
	deviceVector.push_back(cudaDevice);

	//creating buffers
	hpx::lcos::future<buffer> fADataBuffer = cudaDevice.create_buffer(
			count * sizeof(double));
	hpx::lcos::future<buffer> fAIndexBuffer = cudaDevice.create_buffer(
			count * sizeof(int));
	hpx::lcos::future<buffer> fAPointerBuffer = cudaDevice.create_buffer(
			m * sizeof(int));

	hpx::lcos::future<buffer> fBBuffer = cudaDevice.create_buffer(
			n * 1 * sizeof(double));
	hpx::lcos::future<buffer> fCBuffer = cudaDevice.create_buffer(
			m * 1 * sizeof(double));
	hpx::lcos::future<buffer> falphaBuffer = cudaDevice.create_buffer(
			sizeof(double));
	hpx::lcos::future<buffer> fmBuffer = cudaDevice.create_buffer(sizeof(int));
	hpx::lcos::future<buffer> fnBuffer = cudaDevice.create_buffer(sizeof(int));
	hpx::lcos::future<buffer> fcountBuffer = cudaDevice.create_buffer(
			sizeof(int));

	buffer ADataBuffer = fADataBuffer.get();
	buffer AIndexBuffer = fAIndexBuffer.get();
	buffer APointerBuffer = fAPointerBuffer.get();

	buffer BBuffer = fBBuffer.get();
	buffer CBuffer = fCBuffer.get();
	buffer alphaBuffer = falphaBuffer.get();
	buffer mBuffer = fmBuffer.get();
	buffer nBuffer = fnBuffer.get();
	buffer countBuffer = fcountBuffer.get();

	data_futures.push_back(
			ADataBuffer.enqueue_write(0, count * sizeof(double), A_data));
	data_futures.push_back(
			AIndexBuffer.enqueue_write(0, count * sizeof(int), A_indices));
	data_futures.push_back(
			APointerBuffer.enqueue_write(0, m * sizeof(int), A_pointers));

	data_futures.push_back(BBuffer.enqueue_write(0, n * sizeof(double), B));
	data_futures.push_back(CBuffer.enqueue_write(0, m * sizeof(double), C));
	data_futures.push_back(mBuffer.enqueue_write(0, sizeof(int), &m));
	data_futures.push_back(nBuffer.enqueue_write(0, sizeof(int), &n));
	data_futures.push_back(countBuffer.enqueue_write(0, sizeof(int), &count));
	data_futures.push_back(
			alphaBuffer.enqueue_write(0, sizeof(double), &alpha));

	args.push_back(ADataBuffer);
	args.push_back(AIndexBuffer);
	args.push_back(APointerBuffer);

	args.push_back(BBuffer);
	args.push_back(CBuffer);
	args.push_back(mBuffer);
	args.push_back(nBuffer);
	args.push_back(countBuffer);
	args.push_back(alphaBuffer);

	//wait for program to build on all devices
	hpx::wait_all(progBuildVector);

	//Synchronize data transfer before new data is written
	hpx::wait_all(args);

	//Synchronize copy to buffer
	hpx::wait_all(data_futures);

	//run the program on the device
#ifdef HPXCL_CUDA_WITH_STREAMS
	kernelFutures.push_back(prog.run(args, "smvp", grid, block, args));
#else
	kernelFutures.push_back(prog.run(args, "smvp", grid, block));
#endif

	//wait for all the kernal futures to return
	hpx::wait_all(kernelFutures);

	//Free Memory
	args.clear();
	
	double* res = CBuffer.enqueue_read_sync<double>(0, m * sizeof(double));
	
	//Printing the end timing result
	time += timer_stop();
	std::cout << time << " ";

	// Validating the result
	std::cout << validateSmvp(A_data, A_indices, A_pointers, B, res, &m, &n, &count, &alpha) << std::endl;
	
	
	cudaFreeHost(A);
	checkCudaError("svmp free A");
	cudaFreeHost(B);
	checkCudaError("svmp free B");
	cudaFreeHost(C);
	checkCudaError("svmp free C");
	cudaFreeHost(A_data);
	checkCudaError("svmp free A_data");
	cudaFreeHost(A_indices);
	checkCudaError("svmp free A_indices");
	cudaFreeHost(A_pointers);
	checkCudaError("svmp free A_pointers");

	return 0;
}
