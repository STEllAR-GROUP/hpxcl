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
                //functions to run CUDA kernels
                static hpx::lcos::future<float>
                calculate_pi_async(hpx::naming::id_type const& gid,int nthreads, int nblocks)
                {
                    typedef server::device::calculate_pi_action action_type;
                    return hpx::async<action_type>(gid,nthreads,nblocks);
                }

                static float calculate_pi_sync(hpx::naming::id_type const& gid,int nthreads, int nblocks)
                {
                    return calculate_pi_async(gid,nthreads,nblocks).get();
                }

                static hpx::lcos::future<int>
                get_device_id_async(hpx::naming::id_type const& gid)
                {
                    typedef server::device::get_device_id_action action_type;
                    return hpx::async<action_type>(gid);
                }

                static int get_device_id_sync(hpx::naming::id_type const& gid)
                {
                    return get_device_id_async(gid).get();
                }

                static hpx::lcos::future<int> get_context_async(hpx::naming::id_type const& gid)
                {
                    typedef server::device::get_context_action action_type;
                    return hpx::async<action_type>(gid);
                }

                static int get_context_sync(hpx::naming::id_type const& gid)
                {
                    return get_context_async(gid).get();
                }

                static hpx::lcos::future<int> wait(hpx::naming::id_type const& gid)
                {
                    return server::device::wait();
                }

                static hpx::lcos::future<void> create_device_ptr_async(hpx::naming::id_type const &gid, size_t byte_count)
                {
                    typedef server::device::create_device_ptr_action action_type;
                    return hpx::async<action_type>(gid, byte_count);
                }

                static void create_device_ptr(hpx::naming::id_type const &gid, size_t byte_count)
                {
                    create_device_ptr_async(gid, byte_count).get();
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

                /*static hpx::lcos::future<void> launch_kernel_async(hpx::naming::id_type const &gid, hpx::cuda::kernel cu_kernel)
                {
                    typedef server::device::launch_kernel_action action_type;
                    return hpx::async<action_type>(gid, cu_kernel);
                }

                static void launch_kernel(hpx::naming::id_type const &gid, hpx::cuda::kernel cu_kernel)
                {
                    launch_kernel_async(gid, cu_kernel).get();
                }*/

            };
        }
    }
}
#endif
