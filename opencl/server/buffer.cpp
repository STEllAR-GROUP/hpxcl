// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/components.hpp>


#include "buffer.hpp"
#include "../tools.hpp"

using hpx::opencl::server::buffer;
using namespace hpx::opencl::server;


CL_FORBID_EMPTY_CONSTRUCTOR(buffer);

// Constructor
buffer::buffer(device* parent_device, size_t size):memory(parent_device, size)
{

};






























typedef hpx::components::managed_component<buffer> buffer_server_type;
HPX_REGISTER_DERIVED_COMPONENT_FACTORY(buffer_server_type, buffer, "memory");



