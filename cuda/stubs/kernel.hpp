// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(KERNEL_3_HPP)
#define KERNEL_3_HPP

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/include/async.hpp>

#include "../server/kernel.hpp"
#include "kernel.hpp"

namespace hpx
{
    namespace cuda
    {
        namespace stubs
        {
            struct kernel
                : hpx::components::stub_base<server::kernel>
            {  

                static hpx::lcos::future<void> set_stream(hpx::naming::id_type const &gid)
                {
                    typedef server::kernel::set_stream_action action_type;
                    return hpx::async<action_type>(gid);
                }

                static void set_stream_sync(hpx::naming::id_type const &gid)
                {
                    set_stream(gid).get();
                }

                static hpx::lcos::future<void> set_grid_dim(hpx::naming::id_type const &gid, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z)
                {
                    typedef server::kernel::set_grid_dim_action action_type;
                    return hpx::async<action_type>(gid, grid_x, grid_y, grid_z);
                }

                static hpx::lcos::future<void> set_block_dim(hpx::naming::id_type const &gid, unsigned int block_x, unsigned int block_y, unsigned int block_z)
                {
                    typedef server::kernel::set_block_dim_action action_type;
                    return hpx::async<action_type>(gid, block_x, block_y, block_z);
                }

                static void set_grid_dim_sync(hpx::naming::id_type const &gid, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z)
                {
                    set_grid_dim(gid, grid_x, grid_y, grid_z).get();
                }

                static void set_block_dim_sync(hpx::naming::id_type const &gid, unsigned int block_x, unsigned int block_y, unsigned int block_z)
                {
                    set_block_dim(gid, block_x, block_y, block_z).get();
                }

                static hpx::lcos::future<void> load_kernel(hpx::naming::id_type const &gid, std::string file_name)
                {
                    typedef server::kernel::load_kernel_action action_type;
                    return hpx::async<action_type>(gid, file_name);
                }

                static hpx::lcos::future<void> load_module(hpx::naming::id_type const &gid, std::string kernel_name)
                {
                    typedef server::kernel::load_module_action action_type;
                    return hpx::async<action_type>(gid, kernel_name);
                }

                static void load_kernel_sync(hpx::naming::id_type const &gid, std::string kernel_name)
                {
                    load_kernel(gid, kernel_name).get();
                }

                static void load_module_sync(hpx::naming::id_type const &gid, std::string file_name)
                {
                    load_module(gid, file_name).get();
                }

                static hpx::lcos::future<hpx::cuda::server::kernel::Dim3> get_grid(hpx::naming::id_type const &gid)
                {
                    typedef server::kernel::get_grid_action action_type;
                    return hpx::async<action_type>(gid);
                }

                static hpx::cuda::server::kernel::Dim3 get_grid_sync(hpx::naming::id_type const &gid)
                {
                    return get_grid(gid).get();
                }

                static hpx::lcos::future<hpx::cuda::server::kernel::Dim3> get_block(hpx::naming::id_type const &gid)
                {
                    typedef server::kernel::get_block_action action_type;
                    return hpx::async<action_type>(gid);
                }

                static hpx::cuda::server::kernel::Dim3 get_block_sync(hpx::naming::id_type const &gid)
                {
                    return get_block(gid).get();
                }

                static hpx::lcos::future<std::string> get_function(hpx::naming::id_type const &gid)
                {
                    typedef server::kernel::get_function_action action_type;
                    return hpx::async<action_type>(gid);
                } 

                static std::string get_function_sync(hpx::naming::id_type const &gid)
                {
                    return get_function(gid).get();
                }

                static hpx::lcos::future<std::string> get_module(hpx::naming::id_type const &gid)
                {
                    typedef server::kernel::get_module_action action_type;
                    return hpx::async<action_type>(gid);
                }

                static std::string get_module_sync(hpx::naming::id_type const &gid)
                {
                    return get_module(gid).get(); 
                }
            };
        }
    }
}
#endif
