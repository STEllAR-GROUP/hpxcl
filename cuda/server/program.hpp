// Copyright (c)		2013 Damond Howard
// 
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PROGRAM_2_HPP)
#define PROGRAM_2_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/get_ptr.hpp>
#include <hpx/include/util.hpp>

#include <cuda.h>

#include  "../fwd_declarations.hpp"

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

 			 	public:
 			 	program();

 			 	program(hpx::naming::id_type device_id, std::string code);

 			 	program(hpx::naming::id_type device_id, hpx::util::serialize_buffer<char> binary);  			
 			
 			 	~program();

 			 	void build();
 				
 				HPX_DEFINE_COMPONENT_ACTION(program, build);

 			};
 		}
 	}
 }

HPX_REGISTER_ACTION_DECLARATION(
	hpx::cuda::server::program::build_action,
	cuda_program_build_action);

 #endif //PROGRAM_2_HPP