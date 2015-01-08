// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/include/util.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/runtime.hpp>

#include <sstream>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

#include "program.hpp"
#include "kernel.hpp"
#include "../kernel.hpp"

using namespace hpx::cuda::server;

program::program(){}

program::program(hpx::naming::id_type device_id, std::string code){}

program::program(hpx::naming::id_type device_id, hpx::util::serialize_buffer<char> binary){}

program::~program(){}

void program::set_source(std::string source)
{
	this->kernel_source = source;
}
hpx::cuda::kernel program::create_kernel(std::string module_name, std::string kernel_name)
{
	typedef hpx::cuda::server::kernel kernel_type;

	hpx::cuda::kernel cu_kernel(
		hpx::components::new_<kernel_type>(hpx::find_here()));
	cu_kernel.load_module_sync(module_name);
	cu_kernel.load_kernel_sync(kernel_name);
	return cu_kernel;
}
/*hpx::cuda::kernel program::create_kernel(std::string kernel_name)
{
	typedef hpx::cuda::server::kernel kernel_type;

	hpx::cuda::kernel cu_kernel(
		hpx::components::new_<kernel_type>(hpx::find_here()));
	//if kernel and module name are the same
	cu_kernel.load_module_sync(kernel_name);
	cu_kernel.load_kernel_sync(kernel_name);
	return cu_kernel;
}*/

//function used to build a gpu architecture specific cuda kernel
void program::build(std::string NVCC_FLAGS)
{
	std::string NVCC = " ";
	std::string kernel_name = " ";
    int nvcc_exit_status = system(
		(std::string(NVCC) + " -ptx " + NVCC_FLAGS + " " + kernel_name
			+ " -o " + kernel_name).c_str()
			);

	if (nvcc_exit_status)
	{
		std::cerr << "ERROR: nvcc exits with status code: " << nvcc_exit_status	<< std::endl;
		exit(1);
	}
}
