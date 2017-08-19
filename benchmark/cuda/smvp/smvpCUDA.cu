// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cuda.h>
#include <iostream>
#include <cmath>

#include "examples/opencl/benchmark_vector/timer.hpp"

//###########################################################################
//Kernels
//###########################################################################

__global__ void smvp(double *A_data, int *A_indices, int *A_pointers, double *B, double *C, int m, int n, int count, double alpha){
	int ROW = blockIdx.x*blockDim.x+threadIdx.x;

	if(ROW<*m){
		int start = A_pointers[ROW];
		int end = (start==*m-1)?(*count):A_pointers[ROW+1];

		double sum = 0;
		for(int i = start;i<end;i++)
		{
			int index = A_indices[i];
			sum += (*alpha) * A_data[i] * B[index];
		}
		C[ROW] = sum;
	}	

}

//###########################################################################
//Main
//###########################################################################

int main(int argc, char*argv[]) {
	
	if (argc != 3) {
		std::cout << "Usage: " << argv[0] << " #m #n";
		exit(1);
	}

	int m,n,i;

	//Initilizing the matrix dimensions
	m = atoi(argv[1]);
	n = atoi(argv[2]);

	double time = 0;
	timer_start();

	double *A, *B, *C, *A_data;
	double *B_dev, *C_dev, *A_dev;
	int *A_indices, *A_pointers;
	int *AIndices_dev, *APointers_dev;

	double alpha;

	//initializing values of alpha and beta
	alpha = 1.0;

	//Malloc Host
	cudaMallocHost((void**) &A, m*n*sizeof( double ));

	int count = 0;
	//Input can be anything sparse
	for (i = 0; i < (m*n); i++) {
		if((i%n) == 0){
			A[i] = (double)(i+1);
			count++;
		}
	}

	/*
	 * Malloc data on host and device
	 */
	//Malloc Host
	cudaMallocHost((void**) &B, n*1*sizeof( double ));
	cudaMallocHost((void**) &C, m*1*sizeof( double ));
	cudaMallocHost((void**) &A_data, count * sizeof( double ));
	cudaMallocHost((void**) &A_indices, count*sizeof( int ));
	cudaMallocHost((void**) &A_pointers, m * sizeof( int ));

	//Malloc Device
	cudaMalloc((void**) &B_dev, n*1*sizeof( double ));
	cudaMalloc((void**) &C_dev, m*1*sizeof( double ));
	cudaMalloc((void**) &A_dev, count * sizeof( double ));
	cudaMalloc((void**) &AIndices_dev, count*sizeof( int ));
	cudaMalloc((void**) &APointers_dev, m * sizeof( int ));

	//printf (" Intializing matrix data \n");
	
	for (i = 0; i < (1*n); i++) {
		B[i] = (double)(-i-1);
	}

	for (i = 0; i < (m*1); i++) {
		C[i] = 0.0;
	}
	
	//Counters for compression
	int data_counter = 0;
	int index_counter = 0;
	int pointer_counter = -1;

	//Compressing Matrix A
	for (i = 0; i < (m*n); i++) {
		if(A[i] != 0)
		{
			A_data[data_counter++] = A[i];
			if(((int)i/n) != pointer_counter)
				A_pointers[++pointer_counter] = index_counter;
			A_indices[index_counter++] = (i%n);
		}
	}

	dim3 blocksize(32,1);
	dim3 gridsize(1+ceil(m / blocksize.x),1);

	/*
	 * Copy data
	 */
	cudaMemcpy(A_dev, A_data,  count*sizeof( double ), cudaMemcpyHostToDevice);
	cudaMemcpy(AIndices_dev, A_indices,  count*sizeof( double ), cudaMemcpyHostToDevice);
	cudaMemcpy(APointers_dev, A_pointers,  m*sizeof( double ), cudaMemcpyHostToDevice);
	cudaMemcpy(B_dev, B,  n*sizeof( double ), cudaMemcpyHostToDevice);
	cudaMemcpy(C_dev, C,  m*sizeof( double ), cudaMemcpyHostToDevice);

	/*
	 * Kernel launch
	 */
	smvp<<<gridsize, blocksize>>>(A_dev, AIndices_dev, APointers_dev, B_dev, C_dev, m, n, count, alpha);
	cudaDeviceSynchronize();

	/*
	 * Copy result back
	 */
	cudaMemcpy(C, C_dev, m*n*sizeof( double ), cudaMemcpyDeviceToHost);

	/*
	 * Free
	 */
	cudaFreeHost(A);
	cudaFreeHost(A_data);
	cudaFreeHost(A_indices);
	cudaFreeHost(A_pointers;
	cudaFreeHost(B);
	cudaFreeHost(C);
	cudaFree(A_dev);
	cudaFree(B_dev);
	cudaFree(C_dev);
	cudaFree(AIndices_dev);
	cudaFree(APointers_dev);

	//Printing the end timing result
    time+=timer_stop();
    std:: cout << time << std::endl;

	return EXIT_SUCCESS;
}