// Copyright (c)		2013 Damond Howard
// 
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_CUDA_SERVER_PROGRAM_HPP_
#define HPX_CUDA_SERVER_PROGRAM_HPP_

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/get_ptr.hpp>
#include <hpx/include/util.hpp>

#include <iostream>
#include <cuda.h>

#include "../fwd_declarations.hpp"
#include "../kernel.hpp"

 namespace hpx
 {
 	namespace cuda
 	{
 		namespace server
 		{
 			class program
 				: public hpx::components::locking_hook<
 					hpx::components::managed_component_base<program>
 					>
 			{
 			 	private:

 			 	boost::shared_ptr<device> parent_device;
 			 	hpx::naming::id_type parent_device_id;
 			 	std::string kernel_source;
 			 	std::string kernel_name;
 			 	CUlinkState cu_link_state;

 			 	public:
 			 	program();

 			 	program(hpx::naming::id_type device_id, std::string code);

 			 	program(hpx::naming::id_type device_id, hpx::serialization::serialize_buffer<char> binary);
 			
 			 	~program();

				hpx::cuda::kernel create_kernel(std::string module_name, std::string kernel_name);
 			 	
 			 	void build(std::string NVCC_FLAGS);

 			 	void set_source(std::string source);

 				HPX_DEFINE_COMPONENT_ACTION(program, build);
 				HPX_DEFINE_COMPONENT_ACTION(program, create_kernel);
 				HPX_DEFINE_COMPONENT_ACTION(program, set_source);

 			};
 		}
 	}
 }

HPX_REGISTER_ACTION_DECLARATION(
	hpx::cuda::server::program::build_action,
	cuda_program_build_action);
HPX_REGISTER_ACTION_DECLARATION(
	hpx::cuda::server::program::create_kernel_action,
	cuda_program_create_kernel_action);
HPX_REGISTER_ACTION_DECLARATION(
	hpx::cuda::server::program::set_source_action,
	cuda_program_set_source_action);

 #endif //PROGRAM_2_HPP
