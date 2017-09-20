// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

extern "C" { __global__ void dgemm(double *A, double *B, double *C, int *m, int *n, int *k, double *alpha, double *beta){
	int ROW = blockIdx.y*blockDim.y+threadIdx.y;
	int COL = blockIdx.x*blockDim.x+threadIdx.x;

	if(ROW<*m && COL<*n){
		double sum = 0;
		for(int i = 0;i<*k;i++)
		{
			sum+=(*alpha) * A[ROW * (*k) + i] * B[i*(*n)+COL];
			}
		C[ROW*(*n)+COL] = sum + (*beta) * C[ROW*(*n)+COL];
	}	

}
}