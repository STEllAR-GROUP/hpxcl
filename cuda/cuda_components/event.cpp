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

#include "server/event.hpp"

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<
    hpx::cuda::server::event>
    cuda_event_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(cuda_event_type,event);

HPX_REGISTER_ACTION(cuda_event_type::wrapped_type::await_action,
	cuda_event_await_action);
HPX_REGISTER_ACTION(cuda_event_type::wrapped_type::finished_action,
	cuda_event_finished_action);
HPX_REGISTER_ACTION(cuda_event_type::wrapped_type::trigger_action,
	cuda_event_trigger_action);
HPX_REGISTER_ACTION(cuda_event_type::wrapped_type::cuda_event_action,
	cuda_event_cuda_event_action);