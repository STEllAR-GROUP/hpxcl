// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(DEVICE_3_HPP)
#define DEVICE_3_HPP

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/include/async.hpp>

#include <iostream>

#include "../server/device.hpp"
#include "../kernel.hpp"
#include "../program.hpp"
#include "../buffer.hpp"

namespace hpx
{
    namespace cuda
    {
        namespace stubs
        {
            struct device
                : hpx::components::stub_base<server::device>
            {
                static void get_cuda_info(hpx::naming::id_type const& gid)
                {
                    typedef server::device::get_cuda_info_action action_type;
                    hpx::apply<action_type>(gid);
                }

                static void set_device(hpx::naming::id_type const& gid,int dev)
                {
                    typedef server::device::set_device_action action_type;
                    hpx::async<action_type>(gid,dev).get();
                }
                
                static std::vector<int> get_all_devices(std::vector<hpx::naming::id_type> localities)
                {
                    int num = 0;
                    std::vector<int> devices;
                    typedef server::device::get_all_devices_action action_type;
                    for (size_t i=0;i<localities.size();i++)
                    {
                        num +=  hpx::async<action_type>(localities[i]).get();
                        for (int i=0;i<num;i++)
                        {
                         devices.push_back(i);
                        }
                    }
                    return devices;
                }

                static hpx::lcos::future<int>
                get_device_id(hpx::naming::id_type const& gid)
                {
                    typedef server::device::get_device_id_action action_type;
                    return hpx::async<action_type>(gid);
                }

                static int get_device_id_sync(hpx::naming::id_type const& gid)
                {
                    return get_device_id(gid).get();
                }

                static hpx::lcos::future<int> get_context(hpx::naming::id_type const& gid)
                {
                    typedef server::device::get_context_action action_type;
                    return hpx::async<action_type>(gid);
                }

                static int get_context_sync(hpx::naming::id_type const& gid)
                {
                    return get_context(gid).get();
                }

                static hpx::lcos::future<int> wait(hpx::naming::id_type const& gid)
                {
                    return server::device::wait();
                }

                static hpx::lcos::future<void> create_device_ptr(hpx::naming::id_type const &gid, size_t byte_count)
                {
                    typedef server::device::create_device_ptr_action action_type;
                    return hpx::async<action_type>(gid, byte_count);
                }

                static void create_device_ptr_sync(hpx::naming::id_type const &gid, size_t byte_count)
                {
                    create_device_ptr(gid, byte_count).get();
                }

                template <typename T>
                static void create_host_ptr(hpx::naming::id_type const &gid, T value, size_t const byte_count)
                {
                    typedef typename server::device::create_host_ptr_action<T> action_type;
                    std::cout << "hello from create_host_ptr stubs" <<std::endl;
                    hpx::async<action_type>(gid, value, byte_count).get();
                }

                template <typename T>
                static void create_host_ptr_non_blocking(hpx::naming::id_type const &gid, T value, size_t const byte_count)
                {
                   typedef typename server::device::create_host_ptr_action<T> action_type;
                   hpx::apply<action_type>(gid, value, byte_count).get();  
                }

                static hpx::lcos::future<void> mem_cpy_h_to_d(hpx::naming::id_type const &gid, unsigned int variable_id)
                {   
                    typedef server::device::mem_cpy_h_to_d_action action_type;
                    return hpx::async<action_type>(gid, variable_id);
                }

                static void mem_cpy_h_to_d_sync(hpx::naming::id_type const &gid, unsigned int variable_id)
                {
                    mem_cpy_h_to_d(gid, variable_id).get();
                }

                static hpx::lcos::future<void> mem_cpy_d_to_h(hpx::naming::id_type const &gid, unsigned int variable_id)
                {
                    typedef server::device::mem_cpy_d_to_h_action action_type;
                    return hpx::async<action_type>(gid, variable_id);
                }

                static void mem_cpy_d_to_h_sync(hpx::naming::id_type const &gid, unsigned int variable_id)
                {
                    mem_cpy_d_to_h(gid, variable_id).get();
                }

                static hpx::lcos::future<void> launch_kernel(hpx::naming::id_type const &gid, hpx::cuda::kernel cu_kernel)
                {
                    typedef server::device::launch_kernel_action action_type;
                    return hpx::async<action_type>(gid, cu_kernel);
                }

                static void launch_kernel_sync(hpx::naming::id_type const &gid, hpx::cuda::kernel cu_kernel)
                {
                    launch_kernel(gid, cu_kernel).get();
                }

                static hpx::lcos::future<void> free(hpx::naming::id_type const &gid)
                {
                    typedef server::device::free_action action_type;
                    return hpx::async<action_type>(gid);
                }

                static void free_sync(hpx::naming::id_type const &gid)
                {
                    free(gid).get();
                }

                static hpx::lcos::future<hpx::cuda::program>
                create_program_with_source(hpx::naming::id_type const &gid, std::string source)
                {
                    typedef server::device::create_program_with_source_action action_type;
                    return hpx::async<action_type>(gid, source);
                }

                static hpx::cuda::program
                create_program_with_source_sync(hpx::naming::id_type const &gid, std::string source)
                {
                    return create_program_with_source(gid, source).get();
                }

                static hpx::lcos::future<hpx::cuda::buffer>  
                create_buffer(hpx::naming::id_type const &gid, size_t size)
                {
                    typedef server::device::create_buffer_action action_type;
                    return hpx::async<action_type>(gid, size);
                }

                static hpx::cuda::buffer create_buffer_sync(hpx::naming::id_type const &gid, size_t size)
                {
                    return create_buffer(gid, size).get();
                }

            };
        }
    }
}
#endif
