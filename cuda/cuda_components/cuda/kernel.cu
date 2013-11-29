#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "kernel.cuh"
#include <iostream>
#include <thrust/version.h>

//kernel definitions
__global__ void test_kernel(long* vals)
{
  vals[threadIdx.x] += 1;
}

__global__ void calculate_pi_kernel(curandState* states,int trials_per_thread,long* num_of_hits)
{
    unsigned int tid = threadIdx.x * blockDim.x * blockIdx.x;
    int cur_num_of_hits = 0;
    int x,y;
    curand_init(1234,tid,0,&states[tid]);
    for(int i=0;i<trials_per_thread;i++)
    {
        x = curand_uniform(&states[tid]);
        y = curand_uniform(&states[tid]);
        cur_num_of_hits += (x*x + y*y < 1.0f);
    }
    num_of_hits[tid] = cur_num_of_hits;

}

//functions that call CUDA kernels
long gpu_num_of_hits(int blocks,int threads,int trials_per_thread)
{
    long host[blocks * threads];
    curandState *devStates;
    long* num_of_hits ;
    long  hits = 0;

    cudaMalloc((void**) &num_of_hits,blocks * threads * sizeof(long));
    cudaMalloc((void**) &devStates,threads * blocks * sizeof(curandState));

    calculate_pi_kernel<<<blocks,threads>>>(devStates,trials_per_thread,num_of_hits);
    cudaMemcpy(host,num_of_hits,blocks * threads * sizeof(long),cudaMemcpyDeviceToHost);

    for(int i=0;i<trials_per_thread;i++)
    {
        hits += host[i];
    }
    return hits;
}

void cuda_test(long int* a)
{
	long int* a_d;
	cudaMalloc(&a_d,1);
	cudaMemcpy(a_d,a,1,cudaMemcpyHostToDevice);
	test_kernel<<<1,1>>>(a_d);
	cudaMemcpy(a,a_d,1,cudaMemcpyDeviceToHost);
	cudaFree(a_d);
}

//CUDA API wrapper functions
int get_devices()
{
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	return device_count;
}

void cuda_malloc(void **devPtr, size_t size)
{
    cudaMalloc(devPtr, size);
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
