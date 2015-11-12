// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "config.hpp"

//###########################################################################
//Kernels
//###########################################################################

extern "C"  template<typename T>
__global__ void logn(size_t count, T* in, T* out) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count;
			i += gridDim.x * blockDim.x) {
		out[i] = LOG(in[i]);
	}
}

extern "C"  template<typename T>
__global__ void expn(size_t count, T* in, T* out) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count;
			i += gridDim.x * blockDim.x) {
		out[i] = EXP(in[i]);
	}
}

extern "C"  template<typename T>
__global__ void add(size_t count, T* in1, T* in2, T* out) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count;
			i += gridDim.x * blockDim.x) {
		out[i] = in1[i] + in2[i];
	}
}

extern "C"  template<typename T>
__global__ void dbl(size_t count, T* in, T* out) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count;
			i += gridDim.x * blockDim.x) {
		out[i] = 2.0 * in[i];
	}
}

extern "C"  template<typename T>
__global__ void mul(size_t count, T* in1, T* in2, T* out) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count;
			i += gridDim.x * blockDim.x) {
		out[i] = in1[i] * in2[i];
	}
}
