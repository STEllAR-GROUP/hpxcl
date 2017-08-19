// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//###########################################################################
//Kernels
//###########################################################################

__kernel void smvp(__global double *A_data,__global int *A_indices, __global int *A_pointers,
__global double *B, __global double *C, __global int *m, __global int *n, __global int *count,
__global double *alpha)
{
	int ROW = get_global_id(0);

	if(ROW<m[0]){
		int start = A_pointers[ROW];
		int end = (start==m[0]-1)?(count[0]):A_pointers[ROW+1];

		double sum = 0;
		for(int i = start;i<end;i++)
		{
			int index = A_indices[i];
			sum += (alpha[0]) * A_data[i] * B[index];
		}
		C[ROW] = sum;
	}
}