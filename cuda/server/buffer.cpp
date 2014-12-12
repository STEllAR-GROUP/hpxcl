// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/locking_hook.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/include/util.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/runtime.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include "buffer.hpp"

using namespace hpx::cuda::server;

buffer::buffer(){}

buffer::buffer(size_t size)
{
    this->arg_buffer_size = size;
}

buffer::~buffer(){}

size_t buffer::size()
{
    return this->arg_buffer_size;
}

void buffer::enqueue_read(size_t offset, size_t size) const
{
	//read a buffer  
}

void buffer::enqueue_write(size_t offset, size_t size, const void* data) const
{
	//either create pinned memory use cuda 6 memory or just do device copies manually
	//buffer memory is on the device
	
}


