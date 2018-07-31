// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

extern "C" { __global__ void stencil(size_t* count, float* in, float* out, float* s) {

	for (size_t i = (blockDim.x * blockIdx.x +1) + threadIdx.x; i < count[0] - 1;
			i += gridDim.x * blockDim.x) {

		out[i] = s[0] * in[i-1] + s[1] * in[i] + s[2] * in[i+1];
	}
}

}

