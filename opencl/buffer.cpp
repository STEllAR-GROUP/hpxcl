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



HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<
                        hpx::opencl::server::buffer> buffer_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(buffer_type, buffer);


