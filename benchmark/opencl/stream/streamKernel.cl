// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//###########################################################################
//Kernels
//###########################################################################

__kernel void STREAM_Copy(__global double* a, __global double *b,__global int *len) {
	int threadIdX = get_local_id(0);
	int workGroupSize = get_global_size(0);
	int idx = get_global_id(0);

	if(idx < *len)
		b[idx] = a[idx];
} 

__kernel void STREAM_Scale(__global double* a,__global double *b,__global double *scale,__global int *len) {
	int threadIdX = get_local_id(0);
	int workGroupSize = get_global_size(0);
	int idx = get_global_id(0);

	if(idx < *len)
		b[idx] = (*scale) * a[idx];
} 

__kernel void STREAM_Add(__global double* a, __global double *b, __global double *c,__global int *len) {
	int threadIdX = get_local_id(0);
	int workGroupSize = get_global_size(0);
	int idx = get_global_id(0); 

	if(idx < *len)
		a[idx] = c[idx] * b[idx];
}

__kernel void STREAM_Triad(__global double* a,__global double *b,__global double *c,__global double *scale,__global int *len) {
	int threadIdX = get_local_id(0);
	int workGroupSize = get_global_size(0);
	int idx = get_global_id(0);

	if(idx < *len)
		a[idx] = c[idx] * (*scale) + b[idx];
}