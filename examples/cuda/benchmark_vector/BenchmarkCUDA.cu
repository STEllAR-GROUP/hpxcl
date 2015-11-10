// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "utils.hpp"
#include <cuda.h>
#include <iostream>
#include <cmath>

#include "examples/opencl/benchmark_vector/timer.hpp"

//#define SINGLE
#define EPS 10e-6
//###########################################################################
//Switching between single precision and double precision
//###########################################################################

#ifdef SINGLE
#define TYPE float
#define LOG logf
#define EXP expf
#else
#define TYPE double
#define LOG log
#define EXP exp
#endif

//###########################################################################
//Kernels
//###########################################################################

template<typename T>
__global__ void logn(size_t count, T* in, T* out) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count;
			i += gridDim.x * blockDim.x) {
		out[i] = LOG(in[i]);
	}
}

template<typename T>
__global__ void expn(size_t count, T* in, T* out) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count;
			i += gridDim.x * blockDim.x) {
		out[i] = EXP(in[i]);
	}
}

template<typename T>
__global__ void add(size_t count, T* in1, T* in2, T* out) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count;
			i += gridDim.x * blockDim.x) {
		out[i] = in1[i] + in2[i];
	}
}

template<typename T>
__global__ void dbl(size_t count, T* in, T* out) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count;
			i += gridDim.x * blockDim.x) {
		out[i] = 2.0 * in[i];
	}
}

template<typename T>
__global__ void mul(size_t count, T* in1, T* in2, T* out) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count;
			i += gridDim.x * blockDim.x) {
		out[i] = in1[i] * in2[i];
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

	size_t count = atoi(argv[1]);

	std::cout << count << " ";

	//Timer
	double data = 0.;


	//Pointer
	TYPE* out;
	TYPE* out_dev;
	TYPE* in1;
	TYPE* in1_dev;
	TYPE* in2;
	TYPE* in2_dev;

	timer_start();
	//Malloc Host
	cudaMallocHost((void**) &out, count * sizeof(TYPE));
	cudaMallocHost((void**) &in1, count * sizeof(TYPE));
	cudaMallocHost((void**) &in2, count * sizeof(TYPE));
	//Malloc Device
	cudaMallocHost((void**) &out_dev, count * sizeof(TYPE));
	cudaMallocHost((void**) &in1_dev, count * sizeof(TYPE));
	cudaMallocHost((void**) &in2_dev, count * sizeof(TYPE));

	//Initialize the data
	fillRandomVector(in1, count);
	fillRandomVector(in2, count);

	//Copy data to the device
	cudaMemcpy(in1_dev, in1, count * sizeof(TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(in2_dev, in2, count * sizeof(TYPE), cudaMemcpyHostToDevice);
	cudaMemcpy(out_dev, in1, count * sizeof(TYPE), cudaMemcpyHostToDevice);

	data = timer_stop();

	//######################################################################
	//Launch kernels
	//######################################################################

	int gridsize = 1;
	int blocksize = 32;

	// 1. logn kernel
	timer_start();
	logn<<<gridsize, blocksize>>>(count, in1_dev, out_dev);
	cudaDeviceSynchronize();
	cudaMemcpy(out, out_dev, count * sizeof(TYPE), cudaMemcpyDeviceToHost);
	std::cout << timer_stop() << " ";
	for (size_t i = 0; i < count; i++) {
		if (!(std::abs(std::log(in1[i]) - out[i]) < EPS))
			std::cout << "Error for logn at " << i <<  std::endl;
	}

	// 2. expn kernel
	timer_start();
	expn<<<gridsize, blocksize>>>(count, in1_dev, out_dev);
	cudaDeviceSynchronize();
	cudaMemcpy(out, out_dev, count * sizeof(TYPE), cudaMemcpyDeviceToHost);
	std::cout << timer_stop() << " ";
	for (size_t i = 0; i < count; i++) {
		if (!(std::abs(std::exp(in1[i]) - out[i]) < EPS))
			std::cout << "Error for expn at " << i << std::endl;
	}

	// 3. dbl kernel
	timer_start();
	dbl<<<gridsize, blocksize>>>(count, in1_dev, out_dev);
	cudaDeviceSynchronize();
	cudaMemcpy(out, out_dev, count * sizeof(TYPE), cudaMemcpyDeviceToHost);
	std::cout << timer_stop() << " ";
	for (size_t i = 0; i < count; i++) {
		if (!(std::abs(2.0 * in1[i] - out[i]) < EPS))
			std::cout << "Error for dbl at " << i << std::endl;
	}

	// 4. add kernel
	timer_start();
	add<<<gridsize, blocksize>>>(count, in1_dev, in2_dev, out_dev);
	cudaDeviceSynchronize();
	cudaMemcpy(out, out_dev, count * sizeof(TYPE), cudaMemcpyDeviceToHost);
	std::cout << timer_stop() << " ";
	for (size_t i = 0; i < count; i++) {
		if (!(std::abs(in1[i] + in2[i] - out[i]) < EPS))
			std::cout << "Error for add at " << i << std::endl;
	}

	// 5. mul kernel
	timer_start();
    mul<<<gridsize, blocksize>>>(count, in1_dev, in2_dev, out_dev);
    cudaDeviceSynchronize();
	cudaMemcpy(out, out_dev, count * sizeof(TYPE), cudaMemcpyDeviceToHost);
	std::cout << timer_stop() << " ";
	for (size_t i = 0; i < count; i++) {
		if (!(std::abs(in1[i] * in2[i] - out[i]) < EPS))
			std::cout << "Error for mul at " << i << std::endl;
	}

	//######################################################################
	//Clean
	//######################################################################

	timer_start();
	cudaFreeHost(in1);
	cudaFreeHost(in2);
	cudaFreeHost(out);

	cudaFree(in1_dev);
	cudaFree(in2_dev);
	cudaFree(out_dev);

	data += timer_stop();

	std::cout << data << std::endl;

	return EXIT_SUCCESS;
}
