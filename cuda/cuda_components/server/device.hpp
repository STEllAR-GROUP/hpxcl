// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(DEVICE_2_HPP)
#define DEVICE_2_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/util.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/version.h>
#include <boost/make_shared.hpp>
#include <string>

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
             : public hpx::components::locking_hook<hpx::components::managed_component_base<device> >
             {
              	 private:
        	  	 unsigned int device_id;
                 unsigned int context_id;
                 CUdevice cu_device;
                 CUcontext cu_context;
                 std::string device_name;
                 cudaDeviceProp props;
              	 
              	 public:
        	 	 //Constructors
        	  	 //one constructor takes no argument
        	  	 //second constructor takes device_id
        	  	 //third constructor takes a device_info struct
        	 	 device()
                 {
                    cuInit(0); //initializes cuda driver API
                    cuDeviceGet(&cu_device,0);
                    cuCtxCreate(&cu_context,0,cu_device);
                    device_name = props.name;
                 }

        	 	 device(int device_id)
        	 	 {
                     cuInit(0);
                     cuDeviceGet(&cu_device,device_id);
                     cuCtxCreate(&cu_context,0,cu_device);
        	 		 this->set_device(device_id);
                     cudaError_t error;
                     error = cudaGetDeviceProperties(&props,device_id);
                     this->device_name = props.name;
        	 	 }
				 ~device()
				 {
                    cuCtxDetach(cu_context);
                 }

                 //cuda device managedment functions

                 int get_device_count()
                 {
                    int device_count = 0;
                    cuDeviceGetCount(&device_count);
                    return device_count;
                 }

                 void set_device(int dev)
                 {
                	this->device_id = dev;
                    CUresult error;
                    error = cuCtxSetCurrent(cu_context);
                 }

                void get_cuda_info()
                {
                    const int kb = 1024;
                    const int mb = kb * kb;

                    std::cout<<"CUDA version:   v"<<CUDART_VERSION<<std::endl;
                    std::cout<<"Thrust version: v"<<THRUST_MAJOR_VERSION<<"."<<THRUST_MINOR_VERSION<<std::endl<<std::endl;

                    int dev_count = this->get_device_count();
                    //daGetDeviceCount(&dev_count);

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
                        cudaError_t error;
                        error = cudaGetDeviceProperties(&props,i);
                        if(error == cudaErrorInvalidDevice)
                        {   
                            std::cout<<"Device does not exist"<<std::endl;
                        }

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

                 int get_device_id()
                 {
                	return this->device_id;
                 }

                 int get_context()
                 {
                    //returns the current CUDA device context
                    //Cuda sets the device context automatically
                    //return this->context_id;
                    //CUcontext context;
                    //cuCtxGetCurrent(&context);
                    return this->context_id;
                 }

                 int /*hpx::cuda::device*/ get_all_devices()
                 {
                	 //return all devices on this locality
                	 int num_devices = get_device_count();
                     return num_devices;
                 }

                 //void wait_for_event(/*CUevent cu_event*/)
                 //{
                 //}

                 static void  do_wait(boost::shared_ptr<hpx::lcos::local::promise<int> > p)
                 {
                    //actual work
                    std::cout << pi(100,100) << endl;
                    p->set_value(0); //notify the waiting hpx thread and return a value
                 }

                 static hpx::lcos::unique_future<int> wait()
                 { 
                    boost::shared_ptr<hpx::lcos::local::promise<int> >();
                        boost::make_shared<hpx::lcos::local::promise<int> >();

                    hpx::util::io_service_pool* pool =
                        hpx::get_runtime().get_thread_pool("io_pool");

                    pool->get_io_service().post(
                        hpx::util::bind(&do_wait, p));
                    return p->get_future();
                 }

                 float calculate_pi(int nthreads,int nblocks)
                 {
                    return pi(nthreads,nblocks);
                 }
                 
                 HPX_DEFINE_COMPONENT_ACTION(device,calculate_pi);
                 HPX_DEFINE_COMPONENT_ACTION(device,get_cuda_info);
                 HPX_DEFINE_COMPONENT_ACTION(device,set_device);
                 HPX_DEFINE_COMPONENT_ACTION(device,get_all_devices);
                 HPX_DEFINE_COMPONENT_ACTION(device,get_device_id);
                 HPX_DEFINE_COMPONENT_ACTION(device,get_context);
                 //HPX_DEFINE_COMPONENT_ACTION(device,wait_for_event);
                 HPX_DEFINE_COMPONENT_ACTION(device,wait);
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
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::get_device_id_action,
    device_get_device_id_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::get_context_action,
    device__get_context_action);
/*HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::wait_for_event_action,
    device_wait_for_event_action);
*/
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::wait_action,
    device_wait_action);

#endif //cuda_device_2_HPP
