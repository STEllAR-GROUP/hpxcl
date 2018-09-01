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
//Error check
//###########################################################################
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//###########################################################################
//Kernels
//###########################################################################

template<typename T>
__global__ void kernel(size_t offset, T* in) {

	size_t i =  threadIdx.x + blockIdx.x * blockDim.x;
	T x = (T)( i + offset) ;
	T s = sinf(x);
	T c = cosf(x);
	in[i] = in[i] + sqrtf(s * s + c * c);

}

//###########################################################################
//Main
//###########################################################################

int main(int argc, char*argv[]) {

	if (argc != 3) {
		std::cout << "Usage: " << argv[0] << " #elements #gpus"  << std::endl;
		exit(1);
	}

	timer_start();

	size_t count = atoi(argv[1]);
    int deviceCount = atoi(argv[2]);
	double time = 0;
	
	int deviceMax;
	gpuErrchk(cudaGetDeviceCount(&deviceMax));  
   
 	if( deviceCount > deviceMax)
    {
        std::cout << "Error: You are using " << deviceCount << " and only " 
            << deviceMax << "are available" << std::endl;
        exit(1);
    }

	const int blockSize = 256;
	const int n = pow(2,count) * 1024 * blockSize ;
	const int streamSize = n / deviceCount;
	const int streamBytes = streamSize * sizeof(TYPE);
	const int bytes = n * sizeof(TYPE);

	std::cout << n << " ";
	
	//Pointer
	TYPE* in;
	TYPE* in_dev[deviceCount];
  
	//Malloc Host
	gpuErrchk(cudaMallocHost((void**) &in, bytes));
	for (size_t i = 0; i < n ; i++)
        in[i] = 0.;

	//Malloc Device
	for(int i = 0; i < deviceCount; i++)
	{
	 gpuErrchk(cudaSetDevice(i));
	 gpuErrchk(cudaMalloc((void**) &in_dev[i], streamBytes));
	}

	//Create streams
	cudaStream_t stream[deviceCount];
	for (int i = 0; i < deviceCount; ++i)
    {
	    gpuErrchk(cudaSetDevice (i));
		gpuErrchk(cudaStreamCreate(&stream[i]));
    }
 
	//Copy data to device
	for (int i = 0; i < deviceCount; ++i) {
		int offset = i * streamSize;
        gpuErrchk(cudaSetDevice(i));
		gpuErrchk(cudaMemcpyAsync(in_dev[i], &in[offset], streamBytes,
				cudaMemcpyHostToDevice,stream[i]));
	}

	//Launch kernels
	for (int i = 0; i < deviceCount; ++i) {
		int offset = i * streamSize;
		gpuErrchk(cudaSetDevice(i));
		kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(offset,
				in_dev[i]);
	}

	//Copy the result back

	for (int i = 0; i < deviceCount; ++i) {
		int offset = i * streamSize;
		gpuErrchk(cudaSetDevice(i));
		gpuErrchk(cudaMemcpyAsync(&in[offset], in_dev[i], streamBytes,
				cudaMemcpyDeviceToHost, stream[i]));
		
	}
    

	for (int i = 0; i < deviceCount; ++i) {
	gpuErrchk(cudaSetDevice (i));
    gpuErrchk(cudaStreamSynchronize(stream[i]));
	gpuErrchk(cudaDeviceSynchronize());
	}

	time += timer_stop();

	//Check the result
	std::cout << checkKernel(in, n) << " ";

	timer_start();

	//Clean
	gpuErrchk(cudaFreeHost(in));

	for (int i = 0; i < deviceCount; ++i)
    {
		gpuErrchk(cudaSetDevice (i));
        gpuErrchk(cudaFree(in_dev[i]));
		gpuErrchk(cudaStreamDestroy(stream[i]));
    }
	std:: cout << time + timer_stop() << std::endl;

	return EXIT_SUCCESS;
}
