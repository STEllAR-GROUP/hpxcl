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

	const int blockSize = 256, nStreams = 4;
	const int n = pow(2,count) * 1024 * blockSize * nStreams;
	const int streamSize = n / nStreams;
	const int streamBytes = streamSize * sizeof(TYPE);
	const int bytes = n * sizeof(TYPE);

	std::cout << n << " ";

	//Pointer
	TYPE* in;
	TYPE* in_dev;

	//Malloc Host
	cudaMallocHost((void**) &in, bytes);
	memset(in, 0, bytes);

	//Malloc Device
	cudaMalloc((void**) &in_dev, bytes);

	//Create streams
	cudaStream_t stream[nStreams];
	for (int i = 0; i < nStreams; ++i)
		cudaStreamCreate(&stream[i]);

	//Copy data to device
	for (int i = 0; i < nStreams; ++i) {
		int offset = i * streamSize;

		cudaMemcpyAsync(&in_dev[offset], &in[offset], streamBytes,
				cudaMemcpyHostToDevice,stream[i]);
	}

	//Launch kernels
	for (int i = 0; i < nStreams; ++i) {
		int offset = i * streamSize;
		kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(offset,
				in_dev);
	}

	//Copy the result back

	for (int i = 0; i < nStreams; ++i) {
		int offset = i * streamSize;
		cudaMemcpyAsync(&in[offset], &in_dev[offset], streamBytes,
				cudaMemcpyDeviceToHost, stream[i]);
	}

	cudaDeviceSynchronize();

	time += timer_stop();

	//Check the result
	std::cout << checkKernel(in, n) << " ";

	timer_start();

	//Clean
	cudaFreeHost(in);
	cudaFree(in_dev);

	for (int i = 0; i < nStreams; ++i)
		cudaStreamDestroy(stream[i]);

	std:: cout << time + timer_stop() << std::endl;

	return EXIT_SUCCESS;
}
