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

namespace hpx
{
    namespace cuda
    {
        namespace stubs
        {
            struct kernel
                : hpx::components::stub_base<server::kernel>
            {  
                static hpx::lcos::future<void> set_context_async(hpx::naming::id_type const &gid)
                {
                    typedef server::kernel::set_context_action action_type;
                    return hpx::async<action_type>(gid);
                }

            	static void set_context(hpx::naming::id_type const &gid)
                {
                    set_context_async(gid).get();
                }

                static hpx::lcos::future<void> set_stream_async(hpx::naming::id_type const &gid)
                {
                    typedef server::kernel::set_stream_action action_type;
                    return hpx::async<action_type>(gid);
                }

                static void set_stream(hpx::naming::id_type const &gid)
                {
                    set_stream_async(gid).get();
                }

                static hpx::lcos::future<void> set_grid_dim_async(hpx::naming::id_type const &gid, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z)
                {
                    typedef server::kernel::set_grid_dim_action action_type;
                    return hpx::async<action_type>(gid, grid_x, grid_y, grid_z);
                }

                static hpx::lcos::future<void> set_block_dim_async(hpx::naming::id_type const &gid, unsigned int block_x, unsigned int block_y, unsigned int block_z)
                {
                    typedef server::kernel::set_block_dim_action action_type;
                    return hpx::async<action_type>(gid, block_x, block_y, block_z);
                }

                static void set_grid_dim(hpx::naming::id_type const &gid, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z)
                {
                    set_grid_dim_async(gid, grid_x, grid_y, grid_z).get();
                }

                static void set_block_dim(hpx::naming::id_type const &gid, unsigned int block_x, unsigned int block_y, unsigned int block_z)
                {
                    set_block_dim_async(gid, block_x, block_y, block_z);
                }

                static hpx::lcos::future<void> load_kernel_async(hpx::naming::id_type const &gid, const std::string &file_name)
                {
                    typedef server::kernel::load_kernel_action action_type;
                    return hpx::async<action_type>(gid, file_name);
                }

                static hpx::lcos::future<void> load_module_async(hpx::naming::id_type const &gid, const std::string &kernel_name)
                {
                    typedef server::kernel::load_module_action action_type;
                    return hpx::async<action_type>(gid, kernel_name);
                }

                static void load_kernel(hpx::naming::id_type const &gid, const std::string &kernel_name)
                {
                    load_kernel_async(gid, kernel_name).get();
                }

                static void load_module(hpx::naming::id_type const &gid, const std::string &file_name)
                {
                    load_module_async(gid, file_name).get();
                }
            };
        }
    }
}
#endif
