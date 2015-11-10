#include "utils.hpp"
#include <cuda.h>
#include <iostream>
#include <cmath>

#define SINGLE
#define EPS 10e-5
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

int main(void) {

	size_t count = 90;

	//Pointer
	TYPE* out;
	TYPE* out_dev;
	TYPE* in1;
	TYPE* in1_dev;
	TYPE* in2;
	TYPE* in2_dev;

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

	//######################################################################
	//Launch kernels
	//######################################################################

	int gridsize = 1;
	int blocksize = 32;

	// 1. logn kernel

	logn<<<gridsize, blocksize>>>(count, in1_dev, out_dev);
	cudaDeviceSynchronize();
	cudaMemcpy(out, out_dev, count * sizeof(TYPE), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < count; i++) {
		if (!(std::abs(std::log(in1[i]) - out[i]) < EPS))
			std::cout << "Error for logn at " << i <<  std::endl;
	}

	// 2. expn kernel

	expn<<<gridsize, blocksize>>>(count, in1_dev, out_dev);
	cudaDeviceSynchronize();
	cudaMemcpy(out, out_dev, count * sizeof(TYPE), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < count; i++) {
		if (!(std::abs(std::exp(in1[i]) - out[i]) < EPS))
			std::cout << "Error for expn at " << i << std::endl;
	}

	// 3. dbl kernel
	dbl<<<gridsize, blocksize>>>(count, in1_dev, out_dev);
	cudaDeviceSynchronize();
	cudaMemcpy(out, out_dev, count * sizeof(TYPE), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < count; i++) {
		if (!(std::abs(2.0 * in1[i] - out[i]) < EPS))
			std::cout << "Error for dbl at " << i << std::endl;
	}

	// 4. add kernel

	add<<<gridsize, blocksize>>>(count, in1_dev, in2_dev, out_dev);
	cudaDeviceSynchronize();
	cudaMemcpy(out, out_dev, count * sizeof(TYPE), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < count; i++) {
		if (!(std::abs(in1[i] + in2[i] - out[i]) < EPS))
			std::cout << "Error for add at " << i << std::endl;
	}

	// 5. mul kernel

    mul<<<gridsize, blocksize>>>(count, in1_dev, in2_dev, out_dev);
    cudaDeviceSynchronize();
	cudaMemcpy(out, out_dev, count * sizeof(TYPE), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < count; i++) {
		if (!(std::abs(in1[i] * in2[i] - out[i]) < EPS))
			std::cout << "Error for mul at " << i << std::endl;
	}

	return EXIT_SUCCESS;
}
