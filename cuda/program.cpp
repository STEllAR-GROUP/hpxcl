// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/program.hpp"

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<
    hpx::cuda::server::program>
    cuda_program_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(cuda_program_type,program);

HPX_REGISTER_ACTION(cuda_program_type::wrapped_type::build_action,
	cuda_program_build_action);
HPX_REGISTER_ACTION(cuda_program_type::wrapped_type::set_source_action, 
	cuda_program_set_source_action);
HPX_REGISTER_ACTION(cuda_program_type::wrapped_type::create_kernel_action,
	cuda_program_create_kernel_action);