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

		is_local = (hpx::get_colocation_id_sync(get_id()) == hpx::find_here());
	}

	/**
	 * \brief Method returns the buffer's size
	 * \return The buffer size
	 */

	hpx::lcos::future<size_t> size() {
		HPX_ASSERT(this->get_gid());
		typedef server::buffer::size_action action_type;
		return hpx::async<action_type>(this->get_gid());

	}

	size_t size_sync() {
		return size().get();

	}

	/**
	 * \brief Method sets the buffer's size
	 * \param size The size of the buffer
	 *
	 * \note Use this methods carefully for extending a buffer. Adaptive buffer
	 * on GPU are mostly perfoming bad.
	 */

	hpx::lcos::future<void> set_size(size_t size) {
		HPX_ASSERT(this->get_gid());
		typedef server::buffer::set_size_action action_type;
		return hpx::async<action_type>(this->get_gid(), size);

	}

	void set_size_sync(size_t size) {
		set_size(size).get();

	}

	/**
	 * \brief Method copy synchronized the data on the attached device to the host
	 * \param offset Offset, where to start copying data
	 * \param size Size of the data on the device
	 * \return Pointer to the data on the host
	 */

	template<typename T>
	T* enqueue_read_sync(size_t offset, size_t size) {

		if (is_local) {

			T* tmp =
					reinterpret_cast<T*>(enqueue_read_local(offset, size).get());

		} else {

			T* tmp = (T*) enqueue_read(offset, size).get().data();
			return tmp;
		}
	}

	/**
	 * \brief Method copy the data on the attached device to the host
	 * \param offset Offset, where to start copying data
	 * \param size Size of the data on the device
	 * \return A future with the serialized data
	 *
	 * \note This method is for accessing data on remote localities.
	 */

	hpx::future<hpx::serialization::serialize_buffer<char>> enqueue_read(
			size_t offset, size_t size) {
		HPX_ASSERT(this->get_gid());

		typedef server::buffer::enqueue_read_action action_type;
		return hpx::async<action_type>(this->get_gid(), offset, size);

	}

	/**
	 * \brief Method copy the data on the attached device to the host
	 * \param offset Offset, where to start copying data
	 * \param size Size of the data on the device
	 * \return A future with the uintptr_t to the data
	 *
	 * \note This method is for accessing data on local localities.
	 */

	hpx::future<uintptr_t> enqueue_read_local(size_t offset, size_t size) {
		HPX_ASSERT(this->get_gid());

		typedef server::buffer::enqueue_read_local_action action_type;
		return hpx::async<action_type>(this->get_gid(), offset, size);

	}

	/**
	 * \brief Method copies the provided data on the attached device memory
	 * \param offset Offset, where to start copying data
	 * \param size Size of the data on the device
	 * \param data Pointer to the data, which is transfered to the device
	 */

	hpx::future<void> enqueue_write(size_t offset, size_t size,
			const void* data) const {
		HPX_ASSERT(this->get_gid());

		if (is_local) {

			typedef server::buffer::enqueue_write_local_action action_type;
			return hpx::async<action_type>(this->get_id(), offset, size,
					reinterpret_cast<uintptr_t>(data));

		} else {

			hpx::serialization::serialize_buffer<char> serializable_data(
					(char*) data, size,
					hpx::serialization::serialize_buffer<char>::init_mode::reference);

			typedef server::buffer::enqueue_write_action action_type;
			return hpx::async<action_type>(this->get_gid(), offset, size,
					serializable_data);
		}

	}

private:
	hpx::naming::id_type device_gid;
	bool is_local;

};
}
}
#endif //BUFFER_1_HPP
