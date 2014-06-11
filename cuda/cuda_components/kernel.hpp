// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(KERNEL_1_HPP)
#define KERNEL_1_HPP

#include <hpx/include/components.hpp>
#include "stubs/kernel.hpp"

namespace hpx
{
    namespace cuda
    {
        class kernel
            : public hpx::components::client_base<
                kernel, stubs::kernel >
        {
            typedef hpx::components::client_base<
                kernel, stubs::kernel>
                base_type;

            public:
                kernel()
                {}

                kernel(hpx::future<hpx::naming::id_type> const& gid)
                : base_type(gid)
                {}

                hpx::lcos::future<void> set_context_async()
                { 
                    BOOST_ASSERT(this->get_gid());
                    return this->base_type::set_context_async(this->get_gid());
                }

                void set_context()
                {
                    BOOST_ASSERT(this->get_gid());
                    this->base_type::set_context(this->get_gid());
                }

                hpx::lcos::future<void> set_stream_async()
                {
                    BOOST_ASSERT(this->get_gid());
                    return this->base_type::set_stream_async(this->get_gid());
                }

                void set_stream()
                {
                    BOOST_ASSERT(this->get_gid());
                    this->base_type::set_stream(this->get_gid());
                }

                hpx::lcos::future<void> set_grid_dim_async(unsigned int grid_x, unsigned int grid_y, unsigned int grid_z)
                {
                    BOOST_ASSERT(this->get_gid());
                    return this->base_type::set_grid_dim_async(this->get_gid(), grid_x, grid_y, grid_z);
                }

                hpx::lcos::future<void> set_block_dim_async(unsigned int block_x, unsigned int block_y, unsigned int block_z)
                {
                    BOOST_ASSERT(this->get_gid());
                    return this->base_type::set_block_dim_async(this->get_gid(), block_x, block_y, block_z);
                }

                void set_grid_dim(unsigned int grid_x, unsigned int grid_y, unsigned int grid_z)
                {
                    BOOST_ASSERT(this->get_gid());
                    this->base_type::set_grid_dim_async(this->get_gid(), grid_x, grid_y, grid_z);
                }

                void set_block_dim(unsigned int block_x, unsigned int block_y, unsigned int block_z)
                {
                    BOOST_ASSERT(this->get_gid());
                    this->base_type::set_block_dim_async(this->get_gid(), block_x, block_y, block_z);
                }

                hpx::lcos::future<void> load_module_async(const std::string &file_name)
                {
                    BOOST_ASSERT(this->get_gid());
                    return this->base_type::load_module_async(this->get_gid(), file_name);
                }

                hpx::lcos::future<void> load_kernel_async(const std::string &file_name)
                {
                    BOOST_ASSERT(this->get_gid());
                    return this->base_type::load_kernel_async(this->get_gid(), file_name);
                }

                void load_module(const std::string &file_name)
                {
                    BOOST_ASSERT(this->get_gid());
                    this->base_type::load_module(this->get_gid(), file_name);
                }

                void load_kernel(const std::string &file_name)
                {
                    BOOST_ASSERT(this->get_gid());
                    this->base_type::load_kernel(this->get_gid(), file_name);
                }

        };
    }
}
#endif //KERNEL_1_HPP
