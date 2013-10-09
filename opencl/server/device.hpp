// Copyright (c)	2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_SERVER_DEVICE_HPP__
#define HPX_OPENCL_SERVER_DEVICE_HPP__

#include <cstdint>

#include <hpx/include/iostreams.hpp>

#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>

#include <CL/cl.h>

#include "../std.hpp"

////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{ namespace server{
	
	////////////////////////////////////////////////////////
	/// This class represents an OpenCL accelerator device.
	///
	class device
	  : public hpx::components::locking_hook<
	  	hpx::components::managed_component_base<device>
	    >
	{
	public:
		// Constructor
		device(){return;HPX_THROW_EXCEPTION(hpx::no_success, "device()",
			 "empty constructor 'device()' not allowed!");}
		device(clx_device_id deviceID);

		//////////////////////////////////////////////////
		// Exposed functionality of this component
		//

		/// 
		clx_device_id test();

	//[opencl_management_action_types
	HPX_DEFINE_COMPONENT_ACTION(device, test);
	//]

	private:
		///////////////////////////////////////////////
		// Private Member Variables
		//
		cl_device_id 	deviceID;
		cl_platform_id 	platformID;

	};
}}}

//[opencl_management_registration_declarations
HPX_REGISTER_ACTION_DECLARATION(
   	hpx::opencl::server::device::test_action,
	opencl_device_test_action);
	
//]



#endif
