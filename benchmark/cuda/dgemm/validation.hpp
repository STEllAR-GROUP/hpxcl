#ifndef HPX_CUDA_VALIDATION_DGEMM_HPP_
#define HPX_CUDA_VALIDATION_DGEMM_HPP_

#include <cstring>
#include <hpxcl/cuda.hpp>

#define EPS 1e-7

int validateDgemm(double* A, double* B, double* C, double alpha, double beta, int n, int m, int k ) {

	double * CTest;
	cudaMallocHost((void**) &CTest, n*m*sizeof( double ));
	//checkCudaError("Malloc CTest");
	std::memset(CTest, 0, sizeof CTest);


	for( size_t i = 0; i < n; i++) {
		for(size_t j = 0; j < m; j++) {
			double sum = 0;

			for(size_t l = 0; l < k; l++) {

				sum+= alpha * A[i * k + l] * B[l*n+j];
			}

			CTest[i*n+j] = sum + beta * CTest[i*n+j];
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

#endif
