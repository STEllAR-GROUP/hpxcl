// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>

#include <hpxcl/cuda.hpp>

#include "examples/opencl/benchmark_vector/timer.hpp"

using namespace hpx::cuda;

//###########################################################################
//Main
//###########################################################################
int main(int argc, char*argv[]) {

if (argc != 3) {
		std::cout << "Usage: " << argv[0] << " #m #n";
		exit(1);
	}

	int m,n,i;

	//Initilizing the matrix dimensions
	m = atoi(argv[1]);
	n = atoi(argv[2]);

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

	//initializing values of alpha and beta
	alpha = 1.0;

	//Malloc Host
	cudaMallocHost((void**) &A, m*n*sizeof( double ));

	int count = 0;
	//Input can be anything sparse
	for (i = 0; i < (m*n); i++) {
		if((i%n) == 0){
			A[i] = (double)(i+1);
			count++;
		}
	}

	cudaMallocHost((void**) &B, n*1*sizeof( double ));
	cudaMallocHost((void**) &C, m*1*sizeof( double ));
	cudaMallocHost((void**) &A_data, count * sizeof( double ));
	cudaMallocHost((void**) &A_indices, count*sizeof( int ));
	cudaMallocHost((void**) &A_pointers, m * sizeof( int ));

	printf (" Intializing matrix data \n");
	
	for (i = 0; i < (1*n); i++) {
		B[i] = (double)(-i-1);
	}

	for (i = 0; i < (m*1); i++) {
		C[i] = 0.0;
	}
	
	//Counters for compression
	int data_counter = 0;
	int index_counter = 0;
	int pointer_counter = -1;

	//Compressing Matrix A
	for (i = 0; i < (m*n); i++) {
		if(A[i] != 0)
		{
			A_data[data_counter++] = A[i];
			if(((int)i/n) != pointer_counter)
				A_pointers[++pointer_counter] = index_counter;
			A_indices[index_counter++] = (i%n);
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

	grid.x = 1+std::ceil(m / block.x);
	grid.y = 1;
	grid.z = 1;

	std::vector<hpx::future<void>> progBuildVector;
	std::vector<program> progVector;
	std::vector<device> deviceVector;

	//Creating the first device found
	device cudaDevice = devices[0];

	//Create a Mandelbrot device program
	program prog = cudaDevice.create_program_with_file("smvp.cu");

	//Compile with the kernal
	std::vector<std::string> flags;
	std::string mode = "--gpu-architecture=compute_";
	mode.append(
		std::to_string(cudaDevice.get_device_architecture_major().get()));
	mode.append(
		std::to_string(cudaDevice.get_device_architecture_minor().get()));

	flags.push_back(mode);

	progBuildVector.push_back(prog.build(flags, "smvp"));
	progVector.push_back(prog);
	deviceVector.push_back(cudaDevice);

	//wait for program to build on all devices
	hpx::wait_all(progBuildVector);

	//creating buffers
	buffer ADataBuffer = cudaDevice.create_buffer(count*sizeof( double ));
	buffer AIndexBuffer = cudaDevice.create_buffer(count*sizeof( int ));
	buffer APointerBuffer = cudaDevice.create_buffer(m*sizeof( int ));

	buffer BBuffer = cudaDevice.create_buffer(n*1*sizeof( double ));
	buffer CBuffer = cudaDevice.create_buffer(m*1*sizeof( double ));
	buffer alphaBuffer = cudaDevice.create_buffer(sizeof(double));
	buffer mBuffer = cudaDevice.create_buffer(sizeof(int));
	buffer nBuffer = cudaDevice.create_buffer(sizeof(int));
	buffer countBuffer = cudaDevice.create_buffer(sizeof(int));


	data_futures.push_back(ADataBuffer.enqueue_write(0, count*sizeof( double ), A_data));
	data_futures.push_back(AIndexBuffer.enqueue_write(0, count*sizeof( int ), A_indices));
	data_futures.push_back(APointerBuffer.enqueue_write(0, m*sizeof( int ), A_pointers));

	data_futures.push_back(BBuffer.enqueue_write(0, n*sizeof( double ), B));
	data_futures.push_back(CBuffer.enqueue_write(0, m*sizeof( double ), C));
	data_futures.push_back(mBuffer.enqueue_write(0, sizeof( int ), &m));
	data_futures.push_back(nBuffer.enqueue_write(0, sizeof( int ), &n));
	data_futures.push_back(countBuffer.enqueue_write(0, sizeof( int ), &count));
	data_futures.push_back(alphaBuffer.enqueue_write(0, sizeof( double ), &alpha));

	//Synchronize copy to buffer
	hpx::wait_all(data_futures);

	args.push_back(ADataBuffer);
	args.push_back(AIndexBuffer);
	args.push_back(APointerBuffer);

	args.push_back(BBuffer);
	args.push_back(CBuffer);
	args.push_back(mBuffer);
	args.push_back(nBuffer);
	args.push_back(countBuffer);
	args.push_back(alphaBuffer);

	//Synchronize data transfer before new data is written
	hpx::wait_all(args);

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
	
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cudaFree(A_data);
	cudaFree(A_indices);
	cudaFree(A_pointers);

	return 0;
}
