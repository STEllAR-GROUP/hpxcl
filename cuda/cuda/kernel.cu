// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

//CUDA Kernels
//test kernel
extern "C" __global__ void kernel1(int *a)
{
   a++;
}

//vector addition kernel 
extern "C" __global__ void 
	vector_add(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < n)
		c[id] = a[id] + b[id];
}
