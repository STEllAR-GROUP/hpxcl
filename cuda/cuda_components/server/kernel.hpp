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

#include  "../fwd_declarations.hpp"

namespace hpx
{
    namespace cuda
    {
        class Dim3
        {
            public:
                unsigned int x,y,z;
                Dim3(const unsigned int _x, const unsigned int _y, const unsigned int _z)
                : x(_x), y(_y), z(_z)
                {}
                Dim3(const unsigned int *_f)
                : x (_f[0]), y(_f[1]), z(_f[2])
                {}
                Dim3(){}
                ~Dim3() {}
        };

        namespace server
        {
            class kernel
                : public hpx::components::locking_hook<
                    hpx::components::managed_component_base<kernel>
                >
            {  
            	private:
            	CUstream cu_stream;
                CUmodule cu_module;
                CUfunction cu_function;
            	hpx::cuda::Dim3 grid,block;
                std::string  kernel_name;
                std::string  module_name;

            	public:

                kernel()
                {}

            	kernel(std::string kernel_name)
            	{
                    this->kernel_name = kernel_name;
                }
                ~kernel()
                {
                    cuModuleUnload(cu_module);
                }
            	
            	void set_stream()
            	{
                    cuStreamCreate(&cu_stream, CU_STREAM_DEFAULT);
                }
                
                void set_grid_dim(unsigned int grid_x, unsigned int grid_y, unsigned int grid_z)
                {
                    this->grid.x = grid_x;
                    this->grid.y = grid_y;
                    this->grid.z = grid_z;
                }

            	void set_block_dim(unsigned int block_x, unsigned int block_y, unsigned int block_z)
            	{
                    this->block.x = block_x;
                    this->block.y = block_y;
                    this->block.z = block_z;
            	}

                void load_module(const std::string &file_name)
                {
                    //std::cout << "module " << file_name << " loaded" << std::endl;
                    this->module_name = file_name;
                    //cuModuleLoad(&cu_module, file_name.c_str());
                }

                void load_kernel(const std::string &kernal_name)
                {
                    //std::cout << "kernel " << kernel_name << " loaded" << std::endl;
                    this->kernel_name = kernel_name;
                   //cuModuleGetFunction(&cu_function, cu_module, kernel_name.c_str());
                }

                std::string get_function()
                {
                    return this->kernel_name;
                }

                std::string get_module()
                {
                    return this->module_name;
                }

                hpx::cuda::Dim3 get_grid()
                {
                    return this->grid;
                }

                hpx::cuda::Dim3 get_block()
                {
                    return this->block;
                }

                //HPX ation definitions
                HPX_DEFINE_COMPONENT_ACTION(kernel, set_stream);
                HPX_DEFINE_COMPONENT_ACTION(kernel, load_module);
                HPX_DEFINE_COMPONENT_ACTION(kernel, load_kernel);
                HPX_DEFINE_COMPONENT_ACTION(kernel, set_grid_dim);
                HPX_DEFINE_COMPONENT_ACTION(kernel, set_block_dim);
                HPX_DEFINE_COMPONENT_ACTION(kernel, get_function);
                HPX_DEFINE_COMPONENT_ACTION(kernel, get_module);
                HPX_DEFINE_COMPONENT_ACTION(kernel, get_grid);
                HPX_DEFINE_COMPONENT_ACTION(kernel, get_block);

            };
        }
    }
}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::kernel::set_stream_action,
    cuda_kernel_set_stream_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::kernel::load_module_action,
    cuda_kernel_load_module_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::kernel::load_kernel_action,
    cuda_kernel_load_kernel_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::kernel::set_grid_dim_action,
    cuda_kernel_set_grid_dim_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::kernel::set_block_dim_action,
    cuda_kernel_set_dim_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::kernel::get_function_action,
    cuda_kernel_get_function_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::kernel::get_module_action,
    cuda_kernel_get_module_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::kernel::get_block_action,
    cuda_kernel_get_block_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::kernel::get_grid_action,
    cuda_kernel_get_grid_action);

#endif
