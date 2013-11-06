// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include "server/buffer.hpp"
#include "buffer.hpp"

#include "server/device.hpp"
#include "device.hpp"


HPX_REGISTER_COMPONENT_MODULE();


// DEVICE
typedef hpx::components::managed_component<
                        hpx::opencl::server::device> device_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(device_type, device);
HPX_REGISTER_ACTION(device_type::wrapped_type::get_event_data_action,
                    device_get_event_data_action);





// BUFFER
typedef hpx::components::managed_component<
                        hpx::opencl::server::buffer> buffer_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(buffer_type, buffer);
HPX_REGISTER_ACTION(buffer_type::wrapped_type::read_action,
                    buffer_read_action);
HPX_REGISTER_ACTION(buffer_type::wrapped_type::write_action,
                    buffer_write_action);





// EVENT
typedef hpx::components::managed_component<
                        hpx::opencl::server::event> event_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(event_type, event);




// PROGRAM
typedef hpx::components::managed_component<
                        hpx::opencl::server::program> program_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(program_type, program);
HPX_REGISTER_ACTION(program_type::wrapped_type::build_action,
                    program_build_aciton);
