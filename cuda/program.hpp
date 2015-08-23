// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#pragma once
#ifndef HPX_CUDA_PROGRAM_HPP_
#define HPX_CUDA_PROGRAM_HPP_

#include <hpx/include/components.hpp>
#include "server/program.hpp"

namespace hpx {
namespace cuda {
class program: public hpx::components::client_base<program, server::program> {
	typedef hpx::components::client_base<program, server::program> base_type;

public:
	program() {
	}

	program(hpx::future<hpx::naming::id_type> && gid) :
			base_type(std::move(gid)) {
	}

	hpx::lcos::future<void> build(std::string NVCC_FLAGS) {
		HPX_ASSERT(this->get_gid());
		typedef server::program::build_action action_type;
		return hpx::async < action_type > (this->get_gid());
	}

	void build_sync(std::string NVCC_FLAGS) {
		HPX_ASSERT(this->get_gid());
		build(NVCC_FLAGS).get();
	}

	hpx::lcos::future<void> create_kernel(std::string module_name,
			std::string kernel_name) {
		HPX_ASSERT(this->get_gid());
		typedef server::program::create_kernel_action action_type;
		hpx::async<action_type>(this->get_gid());
	}

	void create_kernel_sync(std::string module_name, std::string kernel_name) {
		HPX_ASSERT(this->get_gid());
		create_kernel(this->get_gid(), module_name, kernel_name).get();
	}

	void set_source_sync(std::string source) {
		HPX_ASSERT(this->get_gid());
		 typedef server::program::set_source_action action_type;
		 hpx::async<action_type>(this->get_gid()).get();
	}
};
}
}
#endif //PROGRAM_1_HPP
