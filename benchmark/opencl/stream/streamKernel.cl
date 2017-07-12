// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//###########################################################################
//Kernels
//###########################################################################

__kernel void STREAM_Copy(TYPE* a, TYPE *b, int len) {
	int threadIdX = get_local_id(0);
	int workGroupSize = get_global_size(0);
	int idx = threadIdX + workGroupSize;

	if(idx < len)
		b[idx] = a[idx];
} 

__kernel void STREAM_Scale(TYPE* a, TYPE *b, TYPE scale, int len) {
	int threadIdX = get_local_id(0);
	int workGroupSize = get_global_size(0);
	int idx = threadIdX + workGroupSize;

	if(idx < len)
		b[idx] = scale * a[idx];
} 

__kernel void STREAM_Add(TYPE* a, TYPE *b, TYPE *c, int len) {
	int threadIdX = get_local_id(0);
	int workGroupSize = get_global_size(0);
	int idx = threadIdX + workGroupSize;

	if(idx < len)
		a[idx] = c[idx] * b[idx];
}

__kernel void STREAM_Triad(TYPE* a, TYPE *b, TYPE *c, TYPE scale, int len) {
	int threadIdX = get_local_id(0);
	int workGroupSize = get_global_size(0);
	int idx = threadIdX + workGroupSize;

	if(idx < len)
		a[idx] = c[idx] * scale + b[idx];
}