#if !defined(DEVICE_2_HPP)
#define DEVICE_2_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <boost/foreach.hpp>

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>

#include <thrust/version.h>

#include "../cuda/kernel.cuh"

namespace hpx
{
    namespace cuda
    {
        namespace server
        {
         //////////////////////////////////////
         ///This class represents a cuda device

         class device
             : public hpx::components::locking_hook<
                 hpx::components::managed_component_base<device>
                 >
             {
                 public:

				 device() {}
                 //cuda device managedment functions

                 typedef uint64_t argument_type;

                 int get_device_count()
                 {
                    int device_count = 0;
                    cudaGetDeviceCount(&device_count);
                    return device_count;
                 }

                 void set_device(int dev)
                 {
                    cudaSetDevice(dev);
                 }

                 void get_cuda_info()
                 {
                    const int kb = 1024;
                    const int mb = kb * kb;

                    std::cout<<"CUDA version:   v"<<CUDART_VERSION<<std::endl;
                    std::cout<<"Thrust version: v"<<THRUST_MAJOR_VERSION<<"."<<THRUST_MINOR_VERSION<<std::endl<<std::endl;

                    int dev_count;
                    cudaGetDeviceCount(&dev_count);

                    if(dev_count <= 0)
                    {
                        std::cout<<"No CUDA devices on the current locality"<<std::endl;
                    }
                    else if(dev_count > 0)
                    {
                        std::cout<<"CUDA Devices: "<<std::endl<<std::endl;
                    }
                    for(int i=0;i<dev_count;++i)
                    {
                        cudaDeviceProp props;
                        cudaGetDeviceProperties(&props,i);

                        std::cout<<i<<": "<< props.name<<": "<<props.major<<"."<<props.minor<<std::endl;
                        std::cout<< "   Global memory:   "<<props.totalGlobalMem / mb<<"mb"<<std::endl;
                        std::cout<<"   Shared memory:   " <<props.sharedMemPerBlock / kb<<"kb"<<std::endl;
                        std::cout<<"   Constant memory: " <<props.totalConstMem / kb<<"kb"<<std::endl;
                        std::cout<<"   Block registers: " <<props.regsPerBlock<<std::endl<<std::endl;

                        std::cout<<"   Warp size:         "<<props.warpSize<<std::endl;
                        std::cout<<"   Threads per block: "<<props.maxThreadsPerBlock<<std::endl;
                        std::cout<<"   Max block dimensions: [ " << props.maxThreadsDim[0]<<", "<<props.maxThreadsDim[1]<<", "<<props.maxThreadsDim[2]<<" ]"<<std::endl;
                        std::cout<<"   Max grid dimensions:  [ " << props.maxGridSize[0]<<", "<<props.maxGridSize[1]<< ", "<<props.maxGridSize[2]<<" ]"<<std::endl;
                        std::cout<<std::endl;
                    }
                 }

                 float calculate_pi(int nthreads,int nblocks)
                 {
                    return pi(nthreads,nblocks);
                 }

                 int get_all_devices()
                 {
                    return 0;
                 }

                 template <typename T>
                 T* device_malloc(size_t mem_size)
                 {
                    T* loc = NULL;
                    const int space = mem_size*sizeof(T);
                    cudaError_t error;
                    error = cudaMalloc(&loc,space);
                    if(error == cudaErrorMemoryAllocation)
                    {
                        std::cout<<"Error with memory allocation"<<std::endl;
                    }
                    else if(error == cudaSuccess)
                    {
                        return loc;
                    }
                 }

                 //private:  //All private data members of a cuda device

                 HPX_DEFINE_COMPONENT_ACTION(device,calculate_pi);
                 HPX_DEFINE_COMPONENT_ACTION(device,get_cuda_info);
                 HPX_DEFINE_COMPONENT_ACTION(device,set_device);
                 HPX_DEFINE_COMPONENT_ACTION(device,get_all_devices);

                 template <typename T>
                 struct device_malloc_action
                    : hpx::actions::make_action<T* (device::*)(T),
                        &device::template device_malloc<T*>, device_malloc_action<T*> >
                 {};

            };
	    }
    }
}

//HPX action declarations

HPX_REGISTER_ACTION_DECLARATION(
	hpx::cuda::server::device::calculate_pi_action,
	device_calculate_pi_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::get_cuda_info_action,
    device_get_cuda_info_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::set_device_action,
    device_set_device_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::get_all_devices_action,
    device_get_all_devices_action);
HPX_REGISTER_ACTION_DECLARATION_TEMPLATE(
    (template <typename T>),
    (hpx::cuda::server::device::device_malloc_action<T*>)
    );

#endif //cuda_device_2_HPP
