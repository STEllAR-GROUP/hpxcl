// Copyright (c)    2013 Damond Howard
//                  2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include <hpx/hpx.hpp>

#include "cuda/device.hpp"

typedef hpx::components::managed_component<hpx::cuda::server::device>
    cuda_device_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(cuda_device_type, cuda_device);

HPX_REGISTER_ACTION(cuda_device_type::wrapped_type::get_cuda_info_action,
                    cuda_device_get_cuda_info_action);
HPX_REGISTER_ACTION(
    cuda_device_type::wrapped_type::get_extended_cuda_info_action,
    cuda_device_get_extended_cuda_info_action);
HPX_REGISTER_ACTION(
    cuda_device_type::wrapped_type::get_device_architecture_major_action,
    cuda_device_get_device_architecture_major_action);
HPX_REGISTER_ACTION(
    cuda_device_type::wrapped_type::get_device_architecture_minor_action,
    cuda_device_get_device_architecture_minor_action);
HPX_REGISTER_ACTION(cuda_device_type::wrapped_type::set_device_action,
                    cuda_device_set_device_action);
HPX_REGISTER_ACTION(cuda_device_type::wrapped_type::get_all_devices_action,
                    cuda_device_get_all_devices_action);
HPX_REGISTER_ACTION(cuda_device_type::wrapped_type::get_device_id_action,
                    cuda_device_get_device_id_aciton);
HPX_REGISTER_ACTION(cuda_device_type::wrapped_type::get_context_action,
                    cuda_device_get_context_action);
HPX_REGISTER_ACTION(
    cuda_device_type::wrapped_type::create_program_with_source_action,
    create_program_with_source_action);
HPX_REGISTER_ACTION(cuda_device_type::wrapped_type::create_buffer_action,
                    create_buffer_action);
