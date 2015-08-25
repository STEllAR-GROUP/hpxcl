// Copyright (c)		2013 Damond Howard
//						2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#ifndef HPX_CUDA_KERNEL_HPP_
#define HPX_CUDA_KERNEL_HPP_

#include <hpx/include/components.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/include/async.hpp>
#include "server/kernel.hpp"

namespace hpx {
namespace cuda {
class kernel: public hpx::components::client_base<kernel, cuda::server::kernel> {
	typedef hpx::components::client_base<kernel, cuda::server::kernel> base_type;

public:
	kernel() {
	}

	kernel(hpx::future<hpx::naming::id_type> && gid) :
			base_type(std::move(gid)) {
	}

	hpx::lcos::future<void> set_stream() {
		HPX_ASSERT(this->get_gid());
		typedef server::kernel::set_stream_action action_type;
		return hpx::async < action_type > (this->get_gid());
	}

	void set_stream_sync() {
		set_stream_sync();
	}

	hpx::lcos::future<void> set_grid_dim(unsigned int grid_x,
			unsigned int grid_y, unsigned int grid_z) {
		HPX_ASSERT(this->get_gid());
		typedef server::kernel::get_grid_action action_type;
		return hpx::async < action_type > (this->get_gid());
	}

	hpx::lcos::future<void> set_block_dim(unsigned int block_x,
			unsigned int block_y, unsigned int block_z) {
		HPX_ASSERT(this->get_gid());
		typedef server::kernel::set_grid_dim_action action_type;
		return hpx::async < action_type > (this->get_gid(),block_x,block_y,block_z);
	}

	void set_grid_dim_sync(unsigned int grid_x, unsigned int grid_y,
			unsigned int grid_z) {
		HPX_ASSERT(this->get_gid());
		set_grid_dim(grid_x, grid_y, grid_z);
	}

	void set_block_dim_sync(unsigned int block_x, unsigned int block_y,
			unsigned int block_z) {
		HPX_ASSERT(this->get_gid());
		set_block_dim_sync(block_x, block_y, block_z);
	}

	hpx::lcos::future<void> load_module(std::string file_name) {
		HPX_ASSERT(this->get_gid());
		typedef server::kernel::load_module_action action_type;
		return hpx::async < action_type > (this->get_gid(),file_name);
	}

	hpx::lcos::future<void> load_kernel(std::string file_name) {
		HPX_ASSERT(this->get_gid());
		typedef server::kernel::load_kernel_action action_type;
		return hpx::async < action_type > (this->get_gid(),file_name);
	}

	void load_module_sync(std::string file_name) {
		load_module_sync(file_name);
	}

	void load_kernel_sync(std::string file_name) {
		load_kernel_sync(file_name);
	}

	hpx::lcos::future<hpx::cuda::server::kernel::Dim3> get_grid() {
		HPX_ASSERT(this->get_gid());
		typedef server::kernel::get_grid_action action_type;
		return hpx::async < action_type > (this->get_gid());
	}

	hpx::cuda::server::kernel::Dim3 get_grid_sync() {
		return get_grid_sync();
	}

	hpx::lcos::future<hpx::cuda::server::kernel::Dim3> get_block() {
		HPX_ASSERT(this->get_gid());
		typedef server::kernel::get_block_action action_type;
		return hpx::async < action_type > (this->get_gid());
	}

	hpx::cuda::server::kernel::Dim3 get_block_sync() {

		return get_block_sync();
	}

	hpx::lcos::future<std::string> get_function() {
		HPX_ASSERT(this->get_gid());
		typedef server::kernel::get_function_action action_type;
		return hpx::async < action_type > (this->get_gid());
	}

	std::string get_function_sync() {
		HPX_ASSERT(this->get_gid());
		return get_function_sync();
	}

	hpx::lcos::future<std::string> get_module() {
		HPX_ASSERT(this->get_gid());
		typedef server::kernel::get_module_action action_type;
		return hpx::async < action_type > (this->get_gid());
	}

	std::string get_module_sync() {

		return get_module_sync();
	}

};
}
}
#endif //KERNEL_1_HPP
