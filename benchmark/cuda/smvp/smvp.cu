// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

extern "C" __global__ void smvp(double *A_data, int *A_indices, int *A_pointers, double *B, double *C, int *m, int *n, int *count, double *alpha){
	int ROW = blockIdx.x*blockDim.x+threadIdx.x;

	if(ROW<*m){
		int start = A_pointers[ROW];
		int end = (start==m-1)?(*count):A_pointers[ROW+1];

		double sum = 0;
		for(int i = start;i<end;i++)
		{
			int index = A_indices[i];
			sum += (*alpha) * A_data[index] * B[index];
		}
		C[ROW] = sum;
	}	

}