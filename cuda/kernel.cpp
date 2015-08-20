// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
//#include <hpx/util/portable_binary_iarchive.hpp>
//#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/kernel.hpp"

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<
    hpx::cuda::server::kernel>
    cuda_kernel_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(cuda_kernel_type,kernel);

HPX_REGISTER_ACTION(cuda_kernel_type::wrapped_type::set_stream_action,
	cuda_kernel_set_stream_action);
HPX_REGISTER_ACTION(cuda_kernel_type::wrapped_type::load_module_action,
	cuda_kernel_load_module_action);
HPX_REGISTER_ACTION(cuda_kernel_type::wrapped_type::load_kernel_action,
	cuda_kernel_load_kernel_action);
HPX_REGISTER_ACTION(cuda_kernel_type::wrapped_type::set_grid_dim_action,
	cuda_kernel_set_diminsions_action);
HPX_REGISTER_ACTION(cuda_kernel_type::wrapped_type::set_block_dim_action,
	cuda_kernel_set_block_dim_action);
HPX_REGISTER_ACTION(cuda_kernel_type::wrapped_type::get_function_action, 
	cuda_kernel_get_function_action);
HPX_REGISTER_ACTION(cuda_kernel_type::wrapped_type::get_module_action,
	cuda_kernel_get_module_action);
HPX_REGISTER_ACTION(cuda_kernel_type::wrapped_type::get_grid_action,
	cuda_kernel_get_grid_action);
HPX_REGISTER_ACTION(cuda_kernel_type::wrapped_type::get_block_action,
	cuda_kernel_get_block_action);
