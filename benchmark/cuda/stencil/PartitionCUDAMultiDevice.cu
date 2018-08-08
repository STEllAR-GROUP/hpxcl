// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cuda.h>
#include <iostream>
#include <cmath>

#include "opencl/benchmark_vector/timer.hpp"

#include "config.hpp"
#include "utils.hpp"

//###########################################################################
//Kernels
//###########################################################################

template<typename T>
__global__ void kernel(size_t offset, T* in) {

	size_t i = offset + threadIdx.x + blockIdx.x * blockDim.x;
	T x = (T) i;
	T s = sinf(x);
	T c = cosf(x);
	in[i] = in[i] + sqrtf(s * s + c * c);

}

//###########################################################################
//Main
//###########################################################################

int main(int argc, char*argv[]) {

	if (argc != 2) {
		std::cout << "Usage: " << argv[0] << " #elements" << std::endl;
		exit(1);
	}

	timer_start();

	size_t count = atoi(argv[1]);
	double time = 0;
	
	int* deviceCount;
	cudaGetDeviceCount(&deviceCount);  	

	const int blockSize = 256;
	const int n = pow(2,count) * 1024 * blockSize * deviceCount;
	const int streamSize = n / deviceCount;
	const int streamBytes = streamSize * sizeof(TYPE);
	const int bytes = n * sizeof(TYPE);

	std::cout << n << " ";
	
	//Pointer
	TYPE* in;
	TYPE* in_dev[deviceCount];

	//Malloc Host
	cudaMallocHost((void**) &in, bytes);
	memset(in, 0, bytes);


	//Malloc Device
	for(int i = 0; i < deviceCount; i++)
	{
	 cudaSetDevice (i);
	 cudaMalloc((void**) &(in_dev[i]), streamBytes);
	}

	
	

	//Create streams
	cudaStream_t stream[nStreams];
	for (int i = 0; i < deviceCount; ++i)
	    cudaSetDevice (i);
		cudaStreamCreate(&stream[i]);

	//Copy data to device
	for (int i = 0; i < deviceCount; ++i) {
		int offset = i * streamSize;
        cudaSetDevice (i);
		cudaMemcpyAsync(&(in_dev[i]), &in[offset], streamBytes,
				cudaMemcpyHostToDevice,stream[i]);
	}

	//Launch kernels
	for (int i = 0; i < deviceCount; ++i) {
		int offset = i * streamSize;
		cudaSetDevice (i);
		kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(offset,
				in_dev[i]);
	}

	//Copy the result back

	for (int i = 0; i < deviceCount; ++i) {
		int offset = i * streamSize;
		cudaSetDevice (i);
		cudaMemcpyAsync(&in[offset], &(in_dev[i]), streamBytes,
				cudaMemcpyDeviceToHost, stream[i]);
		
	}

	for (int i = 0; i < deviceCount; ++i) {
	cudaSetDevice (i);
	cudaDeviceSynchronize();
	}

	time += timer_stop();

	//Check the result
	std::cout << checkKernel(in, n) << " ";

	timer_start();

	//Clean
	cudaFreeHost(in);
	cudaFree(in_dev);

	for (int i = 0; i < deviceCount; ++i)
		cudaSetDevice (i);
		cudaStreamDestroy(stream[i]);

	std:: cout << time + timer_stop() << std::endl;

	return EXIT_SUCCESS;
}
