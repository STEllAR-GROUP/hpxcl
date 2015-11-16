// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


extern "C" __global__ void log_float(size_t* count, float* in, float* out) {

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count[0];
			i += gridDim.x * blockDim.x) {
		out[i] = logf(in[i]);
	}
}

extern "C" __global__ void log_double(size_t* count, double* in, double* out) {

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count[0];
			i += gridDim.x * blockDim.x) {
		out[i] = log(in[i]);
	}
}
