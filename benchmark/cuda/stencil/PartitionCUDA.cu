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
__global__ void stencil(size_t offset, T* in, T* out, T* s) {

	int i = offset + threadIdx.x + (blockIdx.x * blockDim.x + 1);
	out[i] = s[0] * in[i - 1] + s[1] * in[i] + s[2] * in[i + 1];

}

//###########################################################################
//Main
//###########################################################################

int main(int argc, char*argv[]) {

	if (argc != 2) {
		std::cout << "Usage: " << argv[0] << " #elements" << std::endl;
		exit(1);
	}

	size_t count = atoi(argv[1]);

	const int blockSize = 256, nStreams = 4;
	const int n = 4 * 1024 * blockSize * nStreams * count;
	const int streamSize = n / nStreams;
	const int streamBytes = streamSize * sizeof(TYPE);
	const int bytes = n * sizeof(TYPE);

	std::cout << bytes << " ";

	//Pointer
	TYPE* out;
	TYPE* out_dev;
	TYPE* in;
	TYPE* in_dev;
	TYPE* s;
	TYPE* s_dev;

	//Malloc Host
	cudaMallocHost((void**) &out, bytes * sizeof(TYPE));
	cudaMallocHost((void**) &in, bytes * sizeof(TYPE));
	cudaMallocHost((void**) &s, 3 * sizeof(TYPE));
	//Malloc Device
	cudaMallocHost((void**) &out_dev, bytes * sizeof(TYPE));
	cudaMallocHost((void**) &in_dev, bytes * sizeof(TYPE));
	cudaMallocHost((void**) &s_dev, 3 * sizeof(TYPE));

	//Initialize the data
	fillRandomVector(in, n);
	s[0] = 0.5;
	s[1] = 1.;
	s[2] = 0.5;

	//Create streams
	cudaStream_t stream[nStreams];
	for (int i = 0; i < nStreams; ++i)
		cudaStreamCreate(&stream[i]);

	//Copy data to device
	for (int i = 0; i < nStreams; ++i) {
		int offset = i * streamSize;

		cudaMemcpyAsync(&in_dev[offset], &in[offset], streamBytes,
				cudaMemcpyHostToDevice, stream[i]);
	}

	//Launch kernels
	for (int i = 0; i < nStreams; ++i) {
		int offset = i * streamSize;
		stencil<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(offset,
				in_dev, out_dev, s_dev);
	}

	//Clean
	cudaFreeHost(in);
	cudaFreeHost(s);
	cudaFreeHost(out);
	cudaFree(out_dev);
	cudaFree(in_dev);
	cudaFree(s_dev);
	for (int i = 0; i < nStreams; ++i)
		cudaStreamDestroy(stream[i]);

	return EXIT_SUCCESS;
}
