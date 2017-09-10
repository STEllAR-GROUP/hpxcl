// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//###########################################################################
//Kernels
//###########################################################################

__kernel void stencil(__global size_t *count,__global double* in,__global double* out,__global double* s) {
	int workGroupX = get_local_size(0);
	int workIdX = get_group_id(0);
	int threadIdX = get_local_id(0);
	int workGroupSize = get_global_size(0);
	int i;
	
	for(i = (workGroupX * workIdX + 1) + threadIdX; i < *count - 1; i += workGroupSize) {
		out[i] = s[0] * in[i-1] + s[1] * in[i] + s[2] * in[i+1];
	}

}