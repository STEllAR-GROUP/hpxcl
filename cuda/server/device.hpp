// Copyright (c)		2013 Damond Howard
//						2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#ifndef HPX_CUDA_SERVER_DEVICE_HPP_
#define HPX_CUDA_SERVER_DEVICE_HPP_

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/include/util.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/runtime.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/version.h>
#include <boost/make_shared.hpp>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>

//#include  "../fwd_declarations.hpp"
#include  "../kernel.hpp"
//#include  "../buffer.hpp"
#include  "../program.hpp"
//#include "../device.hpp"

namespace hpx
{
    namespace cuda
    {
        namespace server
        {
         struct Device_ptr
         {
            CUdeviceptr device_ptr;
            size_t byte_count;
         };
         template <typename T>
         struct Host_ptr
         {
            T *host_ptr;
            size_t byte_count;
         };

        //// This class represents a cuda device /////////
        class device
            :  public hpx::components::locking_hook<
                    hpx::components::managed_component_base<device>
                    >
            {
              	private:

        	    unsigned int device_id;
                unsigned int context_id;
                CUdevice cu_device;
                CUcontext cu_context;
                std::string device_name;
                cudaDeviceProp props;
                std::vector<Device_ptr> device_ptrs;
                std::vector<Host_ptr<int>> host_ptrs;
                int num_args;
              	public:

        	 	device();

        	 	device(int device_id);

  				~device();

                void free();

                int get_device_count();

                void set_device(int dev);

                void get_cuda_info();

                int get_device_id();

                int get_context();

                int get_all_devices();

                static void do_wait(boost::shared_ptr<hpx::lcos::local::promise<int> > p);

                static hpx::lcos::future<int> wait();

                void create_device_ptr(size_t const byte_count);

                template <typename T>
                void create_host_ptr(T value, size_t const byte_count)
                {
                    Host_ptr<T> temp;
                    temp.host_ptr = (T*)malloc(byte_count);
                    (temp.host_ptr) = value;
                    temp.byte_count = byte_count;
                    host_ptrs.push_back(temp);
                }

                void mem_cpy_h_to_d(unsigned int variable_id);

                void mem_cpy_d_to_h(unsigned int variable_id);

                void launch_kernel(hpx::cuda::kernel cu_kernel);

                hpx::cuda::program create_program_with_source(std::string source);

                hpx::cuda::buffer create_buffer(size_t size);

                HPX_DEFINE_COMPONENT_ACTION(device, get_cuda_info);
                HPX_DEFINE_COMPONENT_ACTION(device, set_device);
                HPX_DEFINE_COMPONENT_ACTION(device, get_all_devices);
                HPX_DEFINE_COMPONENT_ACTION(device, get_device_id);
                HPX_DEFINE_COMPONENT_ACTION(device, get_context);
                HPX_DEFINE_COMPONENT_ACTION(device, wait);
                HPX_DEFINE_COMPONENT_ACTION(device, create_device_ptr);
                HPX_DEFINE_COMPONENT_ACTION(device, mem_cpy_h_to_d);
                HPX_DEFINE_COMPONENT_ACTION(device, mem_cpy_d_to_h);
                HPX_DEFINE_COMPONENT_ACTION(device, launch_kernel);
                HPX_DEFINE_COMPONENT_ACTION(device, free);
                HPX_DEFINE_COMPONENT_ACTION(device, create_program_with_source);
                HPX_DEFINE_COMPONENT_ACTION(device, create_buffer);

                template <typename T>
                struct create_host_ptr_action
                   :  hpx::actions::make_action<void (device::*)(T),
                           &device::template create_host_ptr<T>, create_host_ptr_action<T> >
                    {};
            };
	    }
    }
}
/*
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::get_cuda_info_action,
    device_get_cuda_info_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::set_device_action,
    device_set_device_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::get_all_devices_action,
    device_get_all_devices_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::get_device_id_action,
    device_get_device_id_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::get_context_action,
    device__get_context_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::wait_action,
    device_wait_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::create_device_ptr_action,
    device_create_device_ptr_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::mem_cpy_d_to_h_action,
    device_mem_cpy_h_to_d_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::mem_cpy_h_to_d_action,
    device_mem_cpy_d_to_h_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::launch_kernel_action,
    device_launch_kernel_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::free_action,
    dvice_free_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::create_program_with_source_action,
    device_create_program_with_source_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::create_buffer_action,
    device_create_buffer_action);
*/

#endif //cuda_device_2_HPP
