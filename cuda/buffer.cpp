// Copyright (c)		2013 Damond Howard
//						2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

//#include <hpx/hpx.hpp>
//#include <hpx/runtime/components/component_factory.hpp>
//#include <hpx/util/portable_binary_iarchive.hpp>
//#include <hpx/util/portable_binary_oarchive.hpp>

//#include <boost/serialization/version.hpp>
//#include <boost/serialization/export.hpp>

#include "buffer.hpp"


//HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<
    hpx::cuda::server::buffer>
    cuda_buffer_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(cuda_buffer_type,buffer);

HPX_REGISTER_ACTION(
    cuda_buffer_type::wrapped_type::size_action,
	cuda_buffer_size_action);
HPX_REGISTER_ACTION(
	cuda_buffer_type::wrapped_type::set_size_action,
	cuda_buffer_set_size_action);
HPX_REGISTER_ACTION(
	cuda_buffer_type::wrapped_type::enqueue_read_action,
	cuda_buffer_enqueue_read_action);
HPX_REGISTER_ACTION(
	cuda_buffer_type::wrapped_type::enqueue_write_action,
	cuda_buffer_enqueue_write_action);
