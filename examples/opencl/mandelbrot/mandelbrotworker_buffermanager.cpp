// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mandelbrotworker_buffermanager.hpp"

mandelbrotworker_buffermanager::mandelbrotworker_buffermanager(
    hpx::opencl::device device_, size_t initial_buffer_size, bool verbose_,
    cl_mem_flags memflags_)
    : device(device_), verbose(verbose_), memflags(memflags_) {
  // allocate the initial buffer, to improve runtime speed
  allocate_buffer(initial_buffer_size);
}

hpx::opencl::buffer mandelbrotworker_buffermanager::get_buffer(size_t size) {
  // search for an already allocated buffer of the correct size
  buffer_map_type::iterator it = buffers.find(size);

  // if no buffer is found, allocate a new one
  if (it == buffers.end()) {
    allocate_buffer(size);
    it = buffers.find(size);
  }

  // make sure that we now have a buffer
  BOOST_ASSERT(it != buffers.end());

  // return the buffer
  return it->second;
}

void mandelbrotworker_buffermanager::allocate_buffer(size_t size) {
  if (verbose)
    hpx::cout << "allocating opencl buffer of size " << size << " bytes ..."
              << hpx::endl;

  // make sure no buffer of the given size already exists
  BOOST_ASSERT(buffers.find(size) == buffers.end());

  // allocate a buffer
  hpx::opencl::buffer new_buffer = device.create_buffer(memflags, size);

  // add the buffer to the map
  buffers.insert(std::pair<size_t, hpx::opencl::buffer>(size, new_buffer));
}
