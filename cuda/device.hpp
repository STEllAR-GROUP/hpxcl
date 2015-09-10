// Copyright (c)    2013 Damond Howard
//                  2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#ifndef HPX_CUDA_DEVICE_HPP_
#define HPX_CUDA_DEVICE_HPP_

#include <hpx/hpx.hpp>

#include "cuda/fwd_declarations.hpp"
#include "cuda/kernel.hpp"
#include "cuda/buffer.hpp"
#include "cuda/program.hpp"
#include "cuda/server/kernel.hpp"
#include "cuda/server/buffer.hpp"
#include "cuda/server/program.hpp"
#include "cuda/server/device.hpp"

namespace hpx {
namespace cuda {

class device: public hpx::components::client_base<device, cuda::server::device> {
	typedef hpx::components::client_base<device, cuda::server::device> base_type;

public:
	device() {
	}

	device(hpx::naming::id_type const& there, int dev) :
			base_type(hpx::new_ < cuda::server::device > (there, dev)) {
	}

	device(hpx::future<hpx::naming::id_type> && gid) :
			base_type(std::move(gid)) {
	}

	void get_cuda_info() {
		HPX_ASSERT(this->get_gid());
		typedef server::device::get_cuda_info_action action_type;
		hpx::apply < action_type > (this->get_gid());
	}

	void get_extended_cuda_info() {
		HPX_ASSERT(this->get_gid());
		typedef server::device::get_extended_cuda_info_action action_type;
		hpx::apply < action_type > (this->get_gid());
	}

	hpx::lcos::future<int> get_device_architecture_major() {
		HPX_ASSERT(this->get_gid());
		typedef server::device::get_device_architecture_major_action action_type;
		return hpx::async < action_type > (this->get_gid());
	}

	hpx::lcos::future<int> get_device_architecture_minor() {
			HPX_ASSERT(this->get_gid());
			typedef server::device::get_device_architecture_minor_action action_type;
			return hpx::async < action_type > (this->get_gid());
		}

	static std::vector<int> get_all_devices(
			std::vector<hpx::naming::id_type> localities) {
		int num = 0;
		std::vector<int> devices;
		typedef server::device::get_all_devices_action action_type;
		for (size_t i = 0; i < localities.size(); i++) {
			num += hpx::async < action_type > (localities[i]).get();
			for (int i = 0; i < num; i++) {
				devices.push_back(i);
			}
		}
		return devices;
	}

	void set_device(int dev) {
		HPX_ASSERT(this->get_gid());
		typedef server::device::set_device_action action_type;
		hpx::async < action_type > (this->get_gid(), dev);
	}

	hpx::lcos::future<int> get_device_id() {
		HPX_ASSERT(this->get_gid());
		typedef server::device::get_device_id_action action_type;
		return hpx::async < action_type > (this->get_gid());
	}

	int get_device_id_sync() {
		HPX_ASSERT(this->get_gid());
		return get_device_id().get();
	}

	hpx::lcos::future<int> get_context() {
		HPX_ASSERT(this->get_gid());
		typedef server::device::get_context_action action_type;
		return hpx::async < action_type > (this->get_gid());
	}

	int get_context_sync() {

		return get_context().get();
	}

	hpx::lcos::future<int> wait() {
		return server::device::wait();
	}

	hpx::lcos::future<void> create_device_ptr(size_t const byte_count) {
		HPX_ASSERT(this->get_gid());
		typedef server::device::create_device_ptr_action action_type;
		return hpx::async < action_type > (this->get_gid(), byte_count);
	}

	void create_device_ptr_sync(size_t const byte_count) {
		create_device_ptr(byte_count).get();
	}

	template<typename T>
	void create_host_ptr(T value, size_t const byte_count) {
		HPX_ASSERT(this->get_gid());
		typedef typename server::device::create_host_ptr_action<T> action_type;
		std::cout << "hello from create_host_ptr stubs" << std::endl;
		hpx::async < action_type > (this->get_gid()).get();
	}

	template<typename T>
	void create_host_ptr_non_blocking(T value, size_t const byte_count) {
		HPX_ASSERT(this->get_gid());
		typedef typename server::device::create_host_ptr_action<T> action_type;
		hpx::apply < action_type > (this->get_gid()).get();
	}

	hpx::lcos::future<void> mem_cpy_h_to_d(unsigned int variable_id) {
		HPX_ASSERT(this->get_gid());
		typedef server::device::mem_cpy_h_to_d_action action_type;
		return hpx::async < action_type > (this->get_gid(), variable_id);
	}

	void mem_cpy_h_to_d_sync(unsigned int variable_id) {
		HPX_ASSERT(this->get_gid());
		mem_cpy_h_to_d(variable_id).get();
	}

	hpx::lcos::future<void> mem_cpy_d_to_h(unsigned int variable_id) {
		HPX_ASSERT(this->get_gid());
		typedef server::device::mem_cpy_d_to_h_action action_type;
		return hpx::async < action_type > (this->get_gid(), variable_id);
	}

	void mem_cpy_d_to_h_sync(unsigned int variable_id) {
		mem_cpy_d_to_h(variable_id).get();
	}

	hpx::lcos::future<void> launch_kernel(hpx::cuda::kernel cu_kernel) {
		HPX_ASSERT(this->get_gid());
		typedef server::device::launch_kernel_action action_type;
		return hpx::async < action_type > (this->get_gid(), cu_kernel);
	}

	void launch_kernel_sync(hpx::cuda::kernel cu_kernel) {
		launch_kernel(cu_kernel).get();
	}

	/*
	 hpx::lcos::future<void> free()
	 {
	 BOOST_ASSERT(this->get_gid());
	 return this->base_type::free(this->get_gid());
	 }
	 */

	hpx::lcos::future<void> free(hpx::naming::id_type const &gid) {
		typedef server::device::free_action action_type;
		return hpx::async < action_type > (gid);
	}

	void free_sync() {
		HPX_ASSERT(this->get_gid());
		free(this->get_gid());
	}

	hpx::cuda::program create_program_with_source_sync(std::string source) {
		return create_program_with_source(source).get();
	}

	hpx::lcos::future<hpx::cuda::program> create_program_with_source(
			std::string source) {
		HPX_ASSERT(this->get_gid());
		typedef server::device::create_program_with_source_action action_type;
		return hpx::async < action_type > (this->get_gid(), source);
	}

	hpx::lcos::future<hpx::cuda::buffer> create_buffer(size_t size) {
		HPX_ASSERT(this->get_gid());
		typedef server::device::create_buffer_action action_type;
		return hpx::async < action_type > (this->get_gid(), size);
	}

	hpx::cuda::buffer create_buffer_sync(size_t size) {
		HPX_ASSERT(this->get_gid());
		return create_buffer(size).get();
	}
};
}
}
#endif //MANAGED_CUDA_COMPONENT_1_HPP
