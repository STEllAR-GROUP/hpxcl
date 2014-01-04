// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt
#if !defined(KERNEL_2_HPP)
#define KERNEL_2_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>

namespace hpx
{
    namespace cuda
    {
        namespace server
        {
            class kernel
                : public hpx::components::locking_hook<
                    hpx::components::managed_component_base<kernel>
                >
            {   //kernel class data members
            	private:
            	cudaStream_t stream;
            	dim3 dimGrid; 
            	dim3 dimBlock;
                unsigned int parent_device_id;
            	unsigned int kernel_id;
                std::string  kernel_name;

                //Kernel class member functions
            	public:
            	kernel(std::string kernel_name,unsigned int parent_device_id)
            	{
                    this->kernel_name = kernel_name;
                    this->parent_device_id = parent_device_id;
                    
                }

                //define kernel class actions
            	void set_context()
            	{
            		//set the context on which to run the kernel
            		//by default it uses the same context as the
            		//device on which it is called
                    cudaSetDevice(this->parent_device_id);
            	}
            	
            	void set_stream()
            	{
            		//set the stream on which to run the kernel
            		//by default it uses the first stream on the 
            		//device, devices can have multiple streams
            		//streams are used to execute multiple kernels
            		//at the same time on the same device
                    cudaStreamCreate(&stream);

            	}
            	
            	void set_diminsions(int gridX,int gridY, int gridZ,
                                    int blockX,int blockY,int blockZ)
            	{
            		//sets the dimensions the kernel uses for execution
                    dimGrid(gridX,gridY,gridZ);
                    dimBlock(blockX,blockY,blockZ);
            	}

                //sets the arguments of the kernel
                /*void set_args(hpx::cuda::buffer args)
                {

                }*/

                //runs the kernel
                /*hpx::cuda::event
                enqueue()
                {

                }*/

                //HPX ation definitions
                HPX_DEFINE_COMPONENT_ACTION(kernel,set_context);
                HPX_DEFINE_COMPONENT_ACTION(kernel,set_stream);
                HPX_DEFINE_COMPONENT_ACTION(kernel,set_diminsions);
                HPX_DEFINE_COMPONENT_ACTION(kernel,set_args);
            };
        }
    }
}
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::kernel::set_context_action,
    kernel_set_context_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::kernel::set_stream_action,
    kernel_set_stream_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::kernel::set_diminsions,
    kernel_set_diminsions_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::kernel::set_args,
    kernel_set_args_action);


//kernel registration declarations
#endif
