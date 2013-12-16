#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "kernel.cuh"
#include <iostream>
#include <thrust/version.h>

////cuda kernel definitions
__global__ void test_kernel(long* vals)
{
  vals[threadIdx.x] += 1;
}

__global__ void calculate_pi_kernel(float *sum, int nbin, float step, int nthreads, int nblocks)
{
   int i;
   float x;
   int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
   for (i=idx; i< nbin; i+=nthreads*nblocks)
    {
	x = (i+0.5)*step;
	sum[idx] += 4.0/(1.0+x*x);
    }
}

//CUDA kernel wrapper functions

void cuda_test(long int* a)
{
   long int* a_d;
   cudaMalloc(&a_d,1);
   cudaMemcpy(a_d,a,1,cudaMemcpyHostToDevice);
   test_kernel<<<1,1>>>(a_d);
   cudaMemcpy(a,a_d,1,cudaMemcpyDeviceToHost);
   cudaFree(a_d);
}
float pi(int nthreads,int nblocks)
{
    const int NBIN = 10000000;
    const int NUM_BLOCK = 30;
    const int NUM_THREAD = 8;
    int tid = 0;
    float pi = 0.0;

    dim3 dimGrid(NUM_BLOCK,1,1);  // Grid dimensions
    dim3 dimBlock(NUM_THREAD,1,1);  // Block dimensions
    float *sumHost, *sumDev;  // Pointer to host & device arrays

    float step = 1.0/NBIN;  // Step size
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

//CUDA API wrapper functions
int get_devices()
{
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	return device_count;
}

void set_device(int dev)
{
    cudaSetDevice(dev);
}

void get_device_info()
{
    const int kb = 1024;
    const int mb = kb * kb;

    std::cout<<"CUDA version:   v"<<CUDART_VERSION<<std::endl;
    std::cout<<"Thrust version: v"<<THRUST_MAJOR_VERSION<<"."<<THRUST_MINOR_VERSION<<std::endl<<std::endl;

    int dev_count;
    cudaGetDeviceCount(&dev_count);
    std::cout<<"CUDA Devices: "<<std::endl<<std::endl;

    for(int i=0;i<dev_count;++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props,i);

        std::cout<<i<<": "<< props.name<<": "<<props.major<<"."<<props.minor<<std::endl;
        std::cout<< "  Global memory:   "<<props.totalGlobalMem / mb<<"mb"<<std::endl;
        std::cout<<"  Shared memory:   " <<props.sharedMemPerBlock / kb<<"kb"<<std::endl;
        std::cout<<"  Constant memory: " <<props.totalConstMem / kb<<"kb"<<std::endl;
        std::cout<<"  Block registers: " <<props.regsPerBlock<<std::endl<<std::endl;

        std::cout<<"  Warp size:         "<<props.warpSize<<std::endl;
        std::cout<<"  Threads per block: "<<props.maxThreadsPerBlock<<std::endl;
        std::cout<<"  Max block dimensions: [ " << props.maxThreadsDim[0]<<", "<<props.maxThreadsDim[1]<<", "<<props.maxThreadsDim[2]<<" ]"<<std::endl;
        std::cout<<"  Max grid dimensions:  [ " << props.maxGridSize[0]<<", "<<props.maxGridSize[1]<< ", "<<props.maxGridSize[2]<<" ]"<<std::endl;
        std::cout<<std::endl;
    }
}

  //Device management functions
