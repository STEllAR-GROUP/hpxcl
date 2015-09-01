// Copyright (c)		2013 Damond Howard
//						2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include <hpx/hpx.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda/buffer.hpp"

namespace hpx { namespace cuda { namespace server
{

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

void buffer::set_size(size_t size)
{
    this->arg_buffer_size = size;
}

void buffer::enqueue_read(size_t offset, size_t size) const
{
    //read a buffer
}

void buffer::enqueue_write(size_t offset, hpx::serialization::serialize_buffer<char> data)
{
    //write to buffer
}

}}}

