// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(DEVICE_1_HPP)
#define DEVICE_1_HPP

#include <hpx/include/components.hpp>
#include "stubs/device.hpp"
#include "kernel.hpp"

namespace hpx
{
    namespace cuda
    {
        class device
            : public hpx::components::client_base<
                device, stubs::device >
        {
            typedef hpx::components::client_base<
                device, stubs::device
                > base_type;

        public:
            device()
            {}    
            
            device(hpx::future<hpx::naming::id_type> && gid)
				: base_type(std::move(gid))
            {}
            
            void get_cuda_info()
            {
                HPX_ASSERT(this->get_gid());
                this->base_type::get_cuda_info(this->get_gid());
            }

            static std::vector<int> get_all_devices(std::vector<hpx::naming::id_type> localities)
            {
                return base_type::get_all_devices(localities);
            }

            void set_device(int dev)
            {
                HPX_ASSERT(this->get_gid());
                this->base_type::set_device(this->get_gid(),dev);
            }

            hpx::lcos::future<float> calculate_pi_async(int nthreads,int nblocks)
            {
                HPX_ASSERT(this->get_gid());
                return this->base_type::calculate_pi_async(this->get_gid(),nthreads,nblocks);
            }

            float calculate_pi_sync(int nthreads,int nblocks)
            {
                HPX_ASSERT(this->get_gid());
                return this->base_type::calculate_pi_sync(this->get_gid(),nthreads,nblocks);
            }

            hpx::lcos::future<int>
            get_device_id_async()
            {
                HPX_ASSERT(this->get_gid());
                return this->base_type::get_device_id_async(this->get_gid());
            }

            int get_device_id()
            {
                HPX_ASSERT(this->get_gid());
                return this->base_type::get_device_id_sync(this->get_gid());
            }

            hpx::lcos::future<int> get_context_async()
            {
                HPX_ASSERT(this->get_gid());
                return this->base_type::get_context_async(this->get_gid());
            }

            int get_context()
            {
                HPX_ASSERT(this->get_gid());
                return this->base_type::get_context_sync(this->get_gid());
            }

            hpx::lcos::future<int> wait()
            {
                HPX_ASSERT(this->get_gid());
                return this->base_type::wait(this->get_gid());
            }

            hpx::lcos::future<void> create_device_ptr_async(size_t const byte_count)
            {
                HPX_ASSERT(this->get_gid());
                return this->base_type::create_device_ptr_async(this->get_gid(), byte_count);
            }

            void create_device_ptr(size_t const byte_count)
            {
                HPX_ASSERT(this->get_gid());
                this->base_type::create_device_ptr(this->get_gid(), byte_count);
            }

            template <typename T>
            void create_host_ptr(T value, size_t const byte_count)
            {
                HPX_ASSERT(this->get_gid());
                this->base_type::create_host_ptr(this->get_gid(), value, byte_count);
            }

            template <typename T>
            void create_host_ptr_non_blocking(T value, size_t const byte_count)
            {
                HPX_ASSERT(this->get_gid());
                this->base_type::create_host_ptr_non_blocking(this->get_gid(), value, byte_count);
            }

            hpx::lcos::future<void> mem_cpy_h_to_d_async(unsigned int variable_id)
            {
                HPX_ASSERT(this->get_gid());
                return this->base_type::mem_cpy_h_to_d_async(this->get_gid(), variable_id);
            }

            void mem_cpy_h_to_d(unsigned int variable_id)
            {
                HPX_ASSERT(this->get_gid());
                this->base_type::mem_cpy_h_to_d(this->get_gid(), variable_id);
            }

            hpx::lcos::future<void> mem_cpy_d_to_h_async(unsigned int variable_id)
            {
                HPX_ASSERT(this->get_gid());
                return this->base_type::mem_cpy_d_to_h_async(this->get_gid(), variable_id);
            }

            void mem_cpy_d_to_h(unsigned int variable_id)
            {
                HPX_ASSERT(this->get_gid());
                this->base_type::mem_cpy_d_to_h(this->get_gid(), variable_id);
            }

            hpx::lcos::future<void> launch_kernel_async(hpx::cuda::kernel cu_kernel)
            {
                BOOST_ASSERT(this->get_gid());
                return this->base_type::launch_kernel_async(this->get_gid(), cu_kernel);
            }

            void launch_kernel(hpx::cuda::kernel cu_kernel)
            {
                BOOST_ASSERT(this->get_gid());
                this->base_type::launch_kernel(this->get_gid(), cu_kernel);
            }
        };
	}
}
#endif //MANAGED_CUDA_COMPONENT_1_HPP
