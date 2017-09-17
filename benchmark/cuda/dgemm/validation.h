// Copyright (c)       2017 Patrick Diehl
// 				       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstring>
#include <cuda.h>

#define EPS 1e-7

int validateDgemm(double* A, double* B, double* C, double alpha, double beta, int n, int m, int k ) {

	double * CTest;
	cudaMallocHost((void**) &CTest, n*m*sizeof( double ));
	
	std::memset(CTest, 0, sizeof CTest);


	for( size_t i = 0; i < n; i++) {
		for(size_t j = 0; j < m; j++) {
			double sum = 0;

			for(size_t l = 0; l < k; l++) {

				sum+= alpha * A[j * k + l] * B[l*n+i];
			}

			CTest[j*n+i] = sum + beta * CTest[j*n+i];
		}
	}


	int success = 1;

	for( size_t i = 0; i < n * m; i++)

		if (std::abs(CTest[i] - C[i]) > EPS){
			success = 0;
			std::cout << "Error at " << i << " " << CTest[i] << " " << C[i] << std::endl;

		}

	cudaFreeHost(CTest);

	return success;
}