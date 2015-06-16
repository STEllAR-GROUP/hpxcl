// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "server/get_devices.hpp"
#include "server/device.hpp"
#include "server/buffer.hpp"
#include "server/program.hpp"
#include "server/kernel.hpp"

HPX_REGISTER_COMPONENT_MODULE();


// DEVICE
typedef hpx::opencl::server::device device_type;
typedef hpx::components::managed_component<device_type> device_component_type;
HPX_REGISTER_COMPONENT(device_component_type, hpx_opencl_device);

HPX_REGISTER_ACTION(device_type::get_device_info_action);
HPX_REGISTER_ACTION(device_type::get_platform_info_action);
HPX_REGISTER_ACTION(device_type::create_buffer_action);
HPX_REGISTER_ACTION(device_type::create_program_with_source_action);
HPX_REGISTER_ACTION(device_type::create_program_with_binary_action);
HPX_REGISTER_ACTION(device_type::release_event_action);
HPX_REGISTER_ACTION(device_type::activate_deferred_event_action);


// BUFFER
typedef hpx::opencl::server::buffer buffer_type;
typedef hpx::components::managed_component<buffer_type> buffer_component_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(buffer_component_type, hpx_opencl_buffer);

HPX_REGISTER_ACTION(buffer_type::size_action);
HPX_REGISTER_ACTION(buffer_type::enqueue_read_action);
HPX_REGISTER_ACTION(buffer_type::enqueue_send_action);
HPX_REGISTER_ACTION(buffer_type::get_parent_device_id_action);


// PROGRAM
typedef hpx::opencl::server::program program_type;
typedef hpx::components::managed_component<program_type> program_component_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(program_component_type, hpx_opencl_program);

HPX_REGISTER_ACTION(program_type::build_action);
HPX_REGISTER_ACTION(program_type::get_binary_action);
HPX_REGISTER_ACTION(program_type::create_kernel_action);


// KERNEL
typedef hpx::opencl::server::kernel kernel_type;
typedef hpx::components::managed_component<kernel_type> kernel_component_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(kernel_component_type, hpx_opencl_kernel);

HPX_REGISTER_ACTION(kernel_type::set_arg_action);


// GLOBAL ACTIONS
HPX_REGISTER_ACTION(hpx::opencl::server::get_devices_action,
                    hpx_opencl_server_get_devices_action);



