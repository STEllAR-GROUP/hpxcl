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
            	cudaStrream_t stream;
            	dim3 dimGrid; 
            	dim3 dimBlock;
            	
                //Kernel class member functions
            	public:
            	kernel()
            	{}
                //define kernel class actions
            	void set_context()
            	{
            		//set the context on which to run the kernel
            		//by default it uses the same context as the
            		//device on which it is called
            	}
            	
            	void set_stream()
            	{
            		//set the stream on which to run the kernel
            		//by default it uses the first stream on the 
            		//device, devices can have multiple streams
            		//streams are used to execute multiple kernels
            		//at the same time on the same device
            	}
            	
            	void set_diminsions()
            	{
            		//sets the dimensions the kernel uses for execution
            	}
            };
        }
    }
}

//kernel registration declarations
#endif
