//  (C) Copyright 2013 Damond Howard
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/managed_cuda_component.hpp"

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<
	cuda_hpx::server::managed_cuda_component
	> managed_cuda_component_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(managed_cuda_component_type,managed_cuda_component);

HPX_REGISTER_ACTION(
	managed_cuda_component_type::wrapped_type::test1_action,
	managed_cuda_component_test1_action);
HPX_REGISTER_ACTION(
	managed_cuda_component_type::wrapped_type::test2_action,
	managed_cuda_component_test2_action);
HPX_REGISTER_ACTION(
    managed_cuda_component_type::wrapped_type::calculate_pi_action,
    managed_cuda_component_calculate_pi_action);
