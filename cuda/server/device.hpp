// Copyright (c)    2013 Damond Howard
//                  2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt
#pragma once
#ifndef HPX_CUDA_SERVER_DEVICE_HPP_
#define HPX_CUDA_SERVER_DEVICE_HPP_

#include <hpx/hpx.hpp>

#include "cuda/lcos/event.hpp"

#include "cuda/fwd_declarations.hpp"
#include "cuda/export_definitions.hpp"
#include "cuda/buffer.hpp"
#include "cuda/program.hpp"
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include <boost/make_shared.hpp>

namespace hpx {
namespace cuda {
namespace server {

struct device_ptr
{
    CUdeviceptr ptr;
    size_t byte_count;
};

template<typename T>
struct host_ptr
{
    T *ptr;
    size_t byte_count;
};

//// This class represents a cuda device /////////
class HPX_CUDA_EXPORT device: public hpx::components::locking_hook<
        hpx::components::managed_component_base<device> > {

public:

    device();

    device(int device_id);

    ~device();

    void free();

    int get_device_count();

    void set_device(int dev);

    void get_cuda_info();

    void get_extended_cuda_info();

    int get_device_id();

    int get_context();

    int get_device_architecture_major();

    int get_device_architecture_minor();

    int get_all_devices();

    static void do_wait(boost::shared_ptr<hpx::lcos::local::promise<int> > p);

    static hpx::future<int> wait();

    void create_device_ptr(size_t const byte_count);

    void release_event(hpx::naming::gid_type gid);

    void activate_deferred_event(hpx::naming::id_type);

    template<typename T>
    void create_host_ptr(T value, size_t const byte_count) {
        host_ptr<T> temp;
        temp.ptr = (T*) malloc(byte_count);
        (temp.ptr) = value;
        temp.byte_count = byte_count;
        host_ptrs.push_back(temp);
    }

    void mem_cpy_h_to_d(unsigned int variable_id);

    void mem_cpy_d_to_h(unsigned int variable_id);

    hpx::cuda::program create_program_with_source(std::string source);

    hpx::cuda::program create_program_with_file(std::string file);

    hpx::cuda::buffer create_buffer(size_t size);


    HPX_DEFINE_COMPONENT_ACTION(device, get_cuda_info);
    HPX_DEFINE_COMPONENT_ACTION(device, get_extended_cuda_info);
    HPX_DEFINE_COMPONENT_ACTION(device, get_device_architecture_major);
    HPX_DEFINE_COMPONENT_ACTION(device, get_device_architecture_minor);
    HPX_DEFINE_COMPONENT_ACTION(device, set_device);
    HPX_DEFINE_COMPONENT_ACTION(device, get_all_devices);
    HPX_DEFINE_COMPONENT_ACTION(device, get_device_id);
    HPX_DEFINE_COMPONENT_ACTION(device, get_context);
    HPX_DEFINE_COMPONENT_ACTION(device, wait);
    HPX_DEFINE_COMPONENT_ACTION(device, create_program_with_source);
    HPX_DEFINE_COMPONENT_ACTION(device, create_buffer);
    HPX_DEFINE_COMPONENT_ACTION(device, release_event);
    HPX_DEFINE_COMPONENT_ACTION(device, activate_deferred_event);


    template<typename T>
    struct create_host_ptr_action
      : hpx::actions::make_action<
            void (device::*)(T), &device::template create_host_ptr<T>,
                create_host_ptr_action<T> >
    {
    };

private:
    unsigned int device_id;
    unsigned int context_id;
    CUdevice cu_device;
    CUcontext cu_context;
    std::string device_name;
    cudaDeviceProp props;
    std::vector<device_ptr> device_ptrs;
    std::vector<host_ptr<int>> host_ptrs;
    int num_args;

    void print2D(std::string name, int * array);
    void print3D(std::string name, int * array);

};
}
}
}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::get_cuda_info_action,
    device_get_cuda_info_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::get_device_architecture_major_action,
	get_device_architecture_major_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::get_device_architecture_minor_action,
	get_device_architecture_minor_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::get_extended_cuda_info_action,
    device_get_extended_cuda_info_action);
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
    hpx::cuda::server::device::create_program_with_source_action,
    device_create_program_with_source_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::create_buffer_action,
    device_create_buffer_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::release_event_action,
   device_release_event_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::device::activate_deferred_event_action,
    activate_deferred_event_action);

#endif //cuda_device_2_HPP
