// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//###########################################################################
//Kernels
//###########################################################################

__kernel void dgemm(const __global double *A, const __global double *B, __global double *C, const int m, const int n, const int k, const double alpha, const double beta){
	int ROW = get_global_id(1);
	int COL = get_global_id(0);

	if(ROW<(n) && COL<(m)){
	double sum = 0.0;
	for(int i = 0;i<k;i++)
		sum+=(alpha) * A[ROW * (k) + i] * B[i*(n)+COL];
	C[ROW*(n)+COL] = sum + (beta) * C[ROW*(n)+COL];
	}
}