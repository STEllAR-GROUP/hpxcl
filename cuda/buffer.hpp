// Copyright (c)    2013 Damond Howard
//                  2015 patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt
#pragma once
#ifndef HPX_CUDA_BUFFER_HPP_
#define HPX_CUDA_BUFFER_HPP_

#include <hpx/hpx.hpp>

#include "cuda/server/buffer.hpp"

namespace hpx {
namespace cuda {

class buffer: public hpx::components::client_base<buffer, cuda::server::buffer> {
	typedef hpx::components::client_base<buffer, cuda::server::buffer> base_type;

public:
	buffer() {
	}

	buffer(hpx::future<hpx::naming::id_type> && gid) :
			base_type(std::move(gid)) {
	}

	hpx::lcos::future<size_t> size() {
		HPX_ASSERT(this->get_gid());
		typedef server::buffer::size_action action_type;
		return hpx::async < action_type > (this->get_gid());

	}

	size_t size_sync() {
		return size().get();

	}

	hpx::lcos::future<void> set_size(size_t size) {
		HPX_ASSERT(this->get_gid());
		typedef server::buffer::set_size_action action_type;
		return hpx::async < action_type > (this->get_gid(), size);

	}

	void set_size_sync(size_t size) {
		set_size(size).get();

	}

	template<typename T>
	T* enqueue_read_sync(size_t offset, size_t size)
	{
		int * tmp = enqueue_read(offset,size).get().data();
		return tmp;
	}

	hpx::future<hpx::serialization::serialize_buffer<int>> enqueue_read(
			size_t offset, size_t size) {
		HPX_ASSERT(this->get_gid());

		typedef server::buffer::enqueue_read_action action_type;
		return hpx::async < action_type > (this->get_gid(), offset, size);

	}

	hpx::future<void> enqueue_write(size_t offset, size_t size, const int* data) const {
		HPX_ASSERT(this->get_gid());

		hpx::serialization::serialize_buffer<int> serializable_data(
				data, size,
				hpx::serialization::serialize_buffer<int>::init_mode::reference);

		typedef server::buffer::enqueue_write_action action_type;
		return hpx::async < action_type > (this->get_gid(), offset, serializable_data);

	}

private:
            hpx::naming::id_type device_gid;
            bool is_local;

};
}
}
#endif //BUFFER_1_HPP
