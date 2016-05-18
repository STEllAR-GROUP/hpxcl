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
__global__ void stencil(size_t count, T* in, T* out, T* s) {
	for (int i = (blockDim.x * blockIdx.x +1) + threadIdx.x; i < count - 1;
			i += gridDim.x * blockDim.x) {

		out[i] = s[0] * in[i-1] + s[1] * in[i] + s[2] * in[i+1];
	}
}

//###########################################################################
//Main
//###########################################################################

int main(int argc, char*argv[]) {

	if(argc != 2)
	{
		std::cout << "Usage: " << argv[0] << " #elements" << std::endl;
		exit(1);
	}

	double data = 0;
	size_t count = atoi(argv[1]);

	std::cout << count << " ";

	//Pointer
	TYPE* out;
	TYPE* out_dev;
	TYPE* in;
	TYPE* in_dev;
	TYPE* s;
	TYPE* s_dev;

	/*
	 * Malloc data on host and device
	 */
	timer_start();
	//Malloc Host
	cudaMallocHost((void**) &out, count * sizeof(TYPE));
	cudaMallocHost((void**) &in, count * sizeof(TYPE));
	cudaMallocHost((void**) &s, 3 * sizeof(TYPE));
	//Malloc Device
	cudaMalloc((void**) &out_dev, count * sizeof(TYPE));
	cudaMalloc((void**) &in_dev, count * sizeof(TYPE));
	cudaMalloc((void**) &s_dev, 3 * sizeof(TYPE));

	//Initialize the data
	fillRandomVector(in, count);
	s[0] = 0.5;
	s[1] = 1.;
	s[2] = 0.5;

	/*
	 * Copy data
	 */

	cudaMemcpy(in_dev, in, count * sizeof(TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(s_dev, s, 3 * sizeof(TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(out_dev, in, count * sizeof(TYPE), cudaMemcpyHostToDevice);

	int gridsize = 1;
	int blocksize = 32;

	/*
	 * Kernel launch
	 */
	stencil<TYPE><<<gridsize, blocksize>>>(count, in_dev, out_dev, s_dev);
	cudaDeviceSynchronize();

	/*
	 * Copy result back
	 */
	cudaMemcpy(out, out_dev, count * sizeof(TYPE), cudaMemcpyDeviceToHost);

	data += timer_stop();

	//Check the result
	std::cout << checkStencil(in,out,s, count) << " ";

	timer_start();
	/*
	 * Free
	 */
	cudaFreeHost(in);
	cudaFreeHost(s);
	cudaFreeHost(out);
	cudaFree(out_dev);
	cudaFree(in_dev);
	cudaFree(s_dev);

	std:: cout << data + timer_stop() << std::endl;

	return EXIT_SUCCESS;
}
