// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstring>
#include <cuda.h>

#define EPS 1e-7

int validateSmvp(double *A_data, int *A_indices, int *A_pointers, double *B, double *C, int *m, int *n, int *count, double *alpha) {

	double * CTest;
	cudaMallocHost((void**) &CTest, m * 1 * sizeof(double));
	
	std::memset(CTest, 0, sizeof CTest);


	for( size_t i = 0; i < m; i++) {
		int start = A_pointers[i];
		int end = (start==m-1)?(count):A_pointers[i+1];

		double sum = 0;
		for(int j = start;j<end;j++)
		{
			int index = A_indices[j];
			sum += (alpha) * A_data[j] * B[index];
		}
		CTest[i] = sum;
	}


	int success = 1;

	for( size_t i = 0; i < m; i++)

		if (std::abs(CTest[i] - C[i]) > EPS){
			success = 0;
			std::cout << "Error at " << i << " " << CTest[i] << " " << C[i] << std::endl;

		}

	cudaFreeHost(CTest);

	return success;
}