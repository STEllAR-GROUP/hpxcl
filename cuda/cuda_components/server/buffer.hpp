
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

#include <cuda_runtime.h>
#include <cuda.h>

#include <thrust/version.h>

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
 			 		
 			 	public:
 			 	buffer()
 			 	{}

 			 	//HPX action definitions
 			 	HPX_DEFINE_COMPONENT_ACTION(buffer,);
 			};
 		}
 	}
 }

 HPX_REGISTER_ACTION_DECLARATION(,);

 //HPX action declarations
 #endif //BUFFER_2_HPP