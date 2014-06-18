// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

//CUDA Kernels

__global__ void calculate_pi_kernel(float *sum, int nbin, float step, int nthreads, int nblocks)
{
   int i;
   float x;
   int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
   for (i=idx; i< nbin; i+=nthreads*nblocks)
    {
	x = (i+0.5f)*step;
	sum[idx] += 4.0f/(1.0f+x*x);
    }
}
__global__ void kernel1()
{
    //this kernel does nothing
}

//CUDA kernel wrapper functions

float pi(int nthreads,int nblocks)
{
    const int NBIN = 10000000;
    const int NUM_BLOCK = 30;
    const int NUM_THREAD = 8;
    int tid = 0;
    float pi = 0.0f;

    dim3 dimGrid(NUM_BLOCK,1,1);  // Grid dimensions
    dim3 dimBlock(NUM_THREAD,1,1);  // Block dimensions
    float *sumHost, *sumDev;  // Pointer to host & device arrays
    
    float step = 1.0f/NBIN;  // Step size
    size_t size = NUM_BLOCK*NUM_THREAD*sizeof(float);  //Array memory size
    sumHost = (float *)malloc(size);  //  Allocate array on host
    cudaMalloc((void **) &sumDev, size);  // Allocate array on device
    // Initialize array in device to 0
    cudaMemset(sumDev, 0, size);
    //declare a stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // Do calculation on device
    calculate_pi_kernel<<<dimGrid, dimBlock,0,stream>>>(sumDev, NBIN, step, NUM_THREAD, NUM_BLOCK); // call CUDA kernel
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    // Retrieve result from device and store it in host array
    cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
    for(tid=0; tid<NUM_THREAD*NUM_BLOCK; tid++)
     pi += sumHost[tid];
    pi *= step;

    free(sumHost);
    cudaFree(sumDev);

    return pi;
}
