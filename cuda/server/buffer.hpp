
// Copyright (c)		2013 Damond Howard
// 
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(BUFFER_2_HPP)
#define BUFFER_2_HPP

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
 			//////////////////////////////////////////////////////////
 			///This class represents a buffer of cuda kernel arguments

 			class buffer
 				: public hpx::components::locking_hook<
 					hpx::components::managed_component_base<buffer>
 					>
 			{
 			 	private:
 			 	size_t arg_buffer_size; 
 			 	public:
 			 	buffer()
 			 	{
 			 		this->arg_buffer_size = (size_t)0;
 			 	}
 			 	buffer(size_t size)
 			 	{
 			 		this->arg_buffer_size = size;
 			 	} 	
 			 	size_t size()
 			 	{
 			 		return this->arg_buffer_size;
 			 	}
 			 	~buffer()
 			 	{
 			 	}

 			 	void push_back(void *arg)
 			 	{
 			 	}

 			 	void load_args()
 			 	{
 			 	}
 			 	//HPX action definitions
 			 	HPX_DEFINE_COMPONENT_ACTION(buffer,size);
 			 	HPX_DEFINE_COMPONENT_ACTION(buffer,push_back);
 			 	HPX_DEFINE_COMPONENT_ACTION(buffer,load_args);
 			};
 		}
 	}
 }

 HPX_REGISTER_ACTION_DECLARATION(
 	hpx::cuda::server::buffer::size_action,
 	buffer_size_action);
 HPX_REGISTER_ACTION_DECLARATION(
 	hpx::cuda::server::buffer::push_back_action,
 	buffer_push_back_action);
 HPX_REGISTER_ACTION_DECLARATION(
 	hpx::cuda::server::buffer::load_args_action,
 	buffer_load_args_action);

 //HPX action declarations
 #endif //BUFFER_2_HPP