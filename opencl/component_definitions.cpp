// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "server/buffer.hpp"
#include "buffer.hpp"

#include "server/device.hpp"
#include "device.hpp"

#include "server/kernel.hpp"
#include "kernel.hpp"

#include "server/event.hpp"
#include "event.hpp"

#include "server/program.hpp"
#include "program.hpp"

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>


HPX_REGISTER_COMPONENT_MODULE();


// DEVICE
typedef hpx::components::managed_component<
                        hpx::opencl::server::device> device_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(device_type, device);
HPX_REGISTER_ACTION(device_type::wrapped_type::create_user_event_action,
                    device_create_user_event_action);
HPX_REGISTER_ACTION(device_type::wrapped_type::get_device_info_action,
                    device_get_device_info_action);





// BUFFER
typedef hpx::components::managed_component<
                        hpx::opencl::server::buffer> buffer_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(buffer_type, buffer);
HPX_REGISTER_ACTION(buffer_type::wrapped_type::read_action,
                    buffer_read_action);
HPX_REGISTER_ACTION(buffer_type::wrapped_type::write_action,
                    buffer_write_action);
HPX_REGISTER_ACTION(buffer_type::wrapped_type::fill_action,
                    buffer_fill_action);
HPX_REGISTER_ACTION(buffer_type::wrapped_type::size_action,
                    buffer_size_action);
HPX_REGISTER_ACTION(buffer_type::wrapped_type::copy_action,
                    buffer_copy_action);


// EVENT
typedef hpx::components::managed_component<
                        hpx::opencl::server::event> event_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(event_type, event);
HPX_REGISTER_ACTION(event_type::wrapped_type::await_action,
                    event_await_action);
HPX_REGISTER_ACTION(event_type::wrapped_type::get_data_action,
                    event_get_data_action);
HPX_REGISTER_ACTION(event_type::wrapped_type::finished_action,
                    event_finished_action);
HPX_REGISTER_ACTION(event_type::wrapped_type::trigger_action,
                    event_trigger_action);




// PROGRAM
typedef hpx::components::managed_component<
                        hpx::opencl::server::program> program_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(program_type, program);
HPX_REGISTER_ACTION(program_type::wrapped_type::build_action,
                    program_build_action);


// KERNEL
typedef hpx::components::managed_component<
                        hpx::opencl::server::kernel> kernel_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(kernel_type, kernel);
HPX_REGISTER_ACTION(kernel_type::wrapped_type::set_arg_action,
                    kernel_set_arg_action);
HPX_REGISTER_ACTION(kernel_type::wrapped_type::enqueue_action,
                    kernel_enqueue_action);




