// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "server/get_devices.hpp"
#include "server/device.hpp"
#include "server/buffer.hpp"

HPX_REGISTER_COMPONENT_MODULE();


// DEVICE
typedef hpx::opencl::server::device device_type;
typedef hpx::components::managed_component<device_type> device_component_type;
HPX_REGISTER_COMPONENT(device_component_type, hpx_opencl_device);

HPX_REGISTER_ACTION(device_type::get_device_info_action);
HPX_REGISTER_ACTION(device_type::get_platform_info_action);
HPX_REGISTER_ACTION(device_type::create_buffer_action);
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

// GLOBAL ACTIONS
HPX_REGISTER_ACTION(hpx::opencl::server::get_devices_action,
                    hpx_opencl_server_get_devices_action);
