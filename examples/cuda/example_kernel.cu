// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

extern "C" { __global__ void sum2(unsigned int* array, unsigned int* count,
		unsigned int* n) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n[0];
			i += gridDim.x * blockDim.x) {
		atomicAdd(&(count[0]), array[i]);
	}
}
}
