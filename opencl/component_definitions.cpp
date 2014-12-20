// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include "device.hpp"
#include "buffer.hpp"


HPX_REGISTER_COMPONENT_MODULE();


// DEVICE
typedef hpx::components::managed_component<
                        hpx::opencl::server::device> device_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(device_type, hpx_opencl_device);


HPX_REGISTER_ACTION(device_type::wrapped_type::get_device_info_action,
                    hpx_opencl_device_get_device_info_action);

HPX_REGISTER_ACTION(device_type::wrapped_type::get_platform_info_action,
                    hpx_opencl_device_get_platform_info_action);

HPX_REGISTER_ACTION(device_type::wrapped_type::create_buffer_action,
                    hpx_opencl_device_create_buffer_action);


// BUFFER
typedef hpx::components::managed_component<
                        hpx::opencl::server::buffer> buffer_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(buffer_type, hpx_opencl_buffer);

HPX_REGISTER_ACTION(buffer_type::wrapped_type::size_action,
                    hpx_opencl_buffer_size_action);

