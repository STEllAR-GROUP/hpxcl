// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)



extern "C" __global__ void logn(size_t* count, float* in, float* out) {

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count[0];
			i += gridDim.x * blockDim.x) {
		out[i] = logf(in[i]);
	}
}

extern "C" __global__ void expn(size_t* count, float* in, float* out) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count[0];
			i += gridDim.x * blockDim.x) {
		out[i] = expf(in[i]);
	}
}

extern "C" __global__ void add(size_t* count, float* in1, float* out,float* in2) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count[0];
			i += gridDim.x * blockDim.x) {
		out[i] = in1[i] + in2[i];
	}
}

extern "C" __global__ void dbl(size_t* count, float* in, float* out) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count[0];
			i += gridDim.x * blockDim.x) {
		out[i] = 2.0 * in[i];
	}
}

extern "C" __global__ void mul(size_t* count, float* in1, float* out, float* in2) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count[0];
			i += gridDim.x * blockDim.x) {
		out[i] = in1[i] * in2[i];
	}
}

