// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cuda.h>
#include <iostream>
#include <cmath>

//###########################################################################
//Kernels
//###########################################################################

template<typename T>
__global__ void dgemm(double *A, double *B, double *C, int *m, int *n, int *k, double *alpha, double *beta){
	int ROW = blockIdx.y*blockDim.y+threadIdx.y;
	int COL = blockIdx.x*blockDim.x+threadIdx.x;

	if(ROW < (*n) && COL < (*m)){
		double sum = 0;
		for(int i = 0;i<*k;i++)
			sum+=(*alpha) * A[ROW * (*k) + i] * B[i*(*n)+COL];
		C[ROW*(*n)+COL] = sum + (*beta) * C[ROW*(*n)+COL];
	}	
}

//###########################################################################
//Main
//###########################################################################

int main(int argc, char*argv[]) {

	double *A, *B, *C;
	double *A_dev, *B_dev, *C_dev;
	int m,n,k,i;
	double alpha, beta;

	//Initilizing the matrix dimensions
	m = 2000;
	n = 1000;
	k = 200;

	//initializing values of alpha and beta
	alpha = 1.0;
	beta = 0.0;

	/*
	 * Malloc data on host and device
	 */
	//Malloc Host
	cudaMallocHost((void**) &A, m*k*sizeof( double ));
	cudaMallocHost((void**) &B, n*k*sizeof( double ));
	cudaMallocHost((void**) &C, m*n*sizeof( double ));

	//Malloc Device
	cudaMalloc((void**) &A_dev, m*k*sizeof( double ));
	cudaMalloc((void**) &B_dev, n*k*sizeof( double ));
	cudaMalloc((void**) &C_dev, m*n*sizeof( double ));
	
	printf (" Intializing matrix data \n\n");
	for (i = 0; i < (m*k); i++) {
		A[i] = (double)(i+1);
	}

	for (i = 0; i < (k*n); i++) {
		B[i] = (double)(-i-1);
	}

	for (i = 0; i < (m*n); i++) {
		C[i] = 0.0;
	}

	dim3 blocksize(32,32);
	dim3 gridsize(1+ceil(m / block.x),1+ceil(n / block.y));

	/*
	 * Copy data
	 */
	cudaMemcpy(A_dev, A,  m*k*sizeof( double ), cudaMemcpyHostToDevice);
	cudaMemcpy(B_dev, B,  n*k*sizeof( double ), cudaMemcpyHostToDevice);
	cudaMemcpy(C_dev, C,  m*n*sizeof( double ), cudaMemcpyHostToDevice);

	/*
	 * Kernel launch
	 */
	stencil<TYPE><<<gridsize, blocksize>>>(A_dev, B_dev, C_dev, &m, &n, &k, &alpha, &beta);
	cudaDeviceSynchronize();

	/*
	 * Copy result back
	 */
	cudaMemcpy(C, C_dev, m*n*sizeof( double ), cudaMemcpyDeviceToHost);

	/*
	 * Free
	 */
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFree(A_dev);
	cudaFree(B_dev);
	cudaFree(C_dev);

	return EXIT_SUCCESS;
}