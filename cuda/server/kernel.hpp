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

#include <cuda.h>

#include "../buffer.hpp"
#include  "../fwd_declarations.hpp"

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
            {  

                public:
                struct Dim3
                {
                    unsigned int x, y, z;
                    template <typename Archive>
                    void serialize(Archive &ar, unsigned int i)
                    {
                        ar &x;
                        ar &y;
                        ar &z;
                    } 
                };

                kernel();

                kernel(std::string kernel_name);

                ~kernel();
                
                void set_stream();

                void set_grid_dim(unsigned int grid_x, unsigned int grid_y, unsigned int grid_z);

                void set_block_dim(unsigned int block_x, unsigned int block_y, unsigned int block_z);

                void load_module(std::string file_name);

                void load_kernel(std::string kernel_name);

                std::string get_function();

                std::string get_module();

                Dim3 get_grid();

                Dim3 get_block();

                void set_arg(size_t size, hpx::cuda::buffer cu_buffer);

                HPX_DEFINE_COMPONENT_ACTION(kernel, set_stream);
                HPX_DEFINE_COMPONENT_ACTION(kernel, load_module);
                HPX_DEFINE_COMPONENT_ACTION(kernel, load_kernel);
                HPX_DEFINE_COMPONENT_ACTION(kernel, set_grid_dim);
                HPX_DEFINE_COMPONENT_ACTION(kernel, set_block_dim);
                HPX_DEFINE_COMPONENT_ACTION(kernel, get_function);
                HPX_DEFINE_COMPONENT_ACTION(kernel, get_module);
                HPX_DEFINE_COMPONENT_ACTION(kernel, get_grid);
                HPX_DEFINE_COMPONENT_ACTION(kernel, get_block);
                HPX_DEFINE_COMPONENT_ACTION(kernel, set_arg);

                private:
                CUstream cu_stream;
                Dim3 grid,block;
                std::string kernel_name;
                std::string module_name;

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
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::kernel::set_arg_action, 
    cuda_kernel_set_arg_action);

#endif