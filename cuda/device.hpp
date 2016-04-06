// Copyright (c)    2013 Damond Howard
//                  2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt
#pragma once
#ifndef HPX_CUDA_DEVICE_HPP_
#define HPX_CUDA_DEVICE_HPP_

#include <hpx/hpx.hpp>

#include "cuda/server/buffer.hpp"
#include "cuda/server/program.hpp"
#include "cuda/server/device.hpp"

namespace hpx {
namespace cuda {

class device: public hpx::components::client_base<device, cuda::server::device> {
	typedef hpx::components::client_base<device, cuda::server::device> base_type;

public:
	device() {
	}

	device(hpx::naming::id_type const& there, int dev) :
			base_type(hpx::new_<cuda::server::device>(there, dev)) {
	}

	device(hpx::future<hpx::naming::id_type> && gid) :
			base_type(std::move(gid)) {
	}

	/**
	 * \brief Method prints the properties of this device
	 */
	void get_cuda_info() {
		HPX_ASSERT(this->get_gid());
		typedef server::device::get_cuda_info_action action_type;
		hpx::apply<action_type>(this->get_gid());
	}

	/**
	 * \brief Method prints the extended properties of this device
	 *
	 * \note All information of the cudaDeviceproperties are shown.
	 */
	void get_extended_cuda_info() {
		HPX_ASSERT(this->get_gid());
		typedef server::device::get_extended_cuda_info_action action_type;
		hpx::apply<action_type>(this->get_gid());
	}

	/**
	 * \brief methods return the major compute capability of this device
	 * \return Major compute capability of this device
	 */
	hpx::lcos::future<int> get_device_architecture_major() {
		HPX_ASSERT(this->get_gid());
		typedef server::device::get_device_architecture_major_action action_type;
		return hpx::async<action_type>(this->get_gid());
	}

	/**
	 * \brief methods return the minor compute capability of this device
	 * \return Minor compute capability of this device
	 */
	hpx::lcos::future<int> get_device_architecture_minor() {
		HPX_ASSERT(this->get_gid());
		typedef server::device::get_device_architecture_minor_action action_type;
		return hpx::async<action_type>(this->get_gid());
	}

	/**
	 * \brief Method returns all other devices on the locality of this device
	 * \return A list of suitable CUDA devices on target node
	 */

	static std::vector<int> get_all_devices(
			std::vector<hpx::naming::id_type> localities) {
		int num = 0;
		std::vector<int> devices;
		typedef server::device::get_all_devices_action action_type;
		for (size_t i = 0; i < localities.size(); i++) {
			num += hpx::async<action_type>(localities[i]).get();
			for (int i = 0; i < num; i++) {
				devices.push_back(i);
			}
		}
		return devices;
	}

	void set_device(int dev) {
		HPX_ASSERT(this->get_gid());
		typedef server::device::set_device_action action_type;
		hpx::async<action_type>(this->get_gid(), dev);
	}

	hpx::lcos::future<int> get_device_id() {
		HPX_ASSERT(this->get_gid());
		typedef server::device::get_device_id_action action_type;
		return hpx::async<action_type>(this->get_gid());
	}

	int get_device_id_sync() {
		HPX_ASSERT(this->get_gid());
		return get_device_id().get();
	}

	hpx::lcos::future<int> get_context() {
		HPX_ASSERT(this->get_gid());
		typedef server::device::get_context_action action_type;
		return hpx::async<action_type>(this->get_gid());
	}

	int get_context_sync() {

		return get_context().get();
	}

	hpx::lcos::future<int> wait() {
		return server::device::wait();
	}

	/**
	 * \brief Creates synchronous a program with the set source code
	 */
	hpx::cuda::program create_program_with_source_sync(std::string source) {
		return create_program_with_source(source).get();
	}

	/**
	 * \brief Creates a program with the set source code
	 *
	 * This Method creates a program containing the source code.
	 * This program is attached to this device and all streams are
	 * executed there.
	 *
	 * \param source The source code of the CUDA kernel
	 *
	 * \see Program
	 */
	hpx::lcos::future<hpx::cuda::program> create_program_with_source(
			std::string source) {
		HPX_ASSERT(this->get_gid());
		typedef server::device::create_program_with_source_action action_type;
		return hpx::async<action_type>(this->get_gid(), source);
	}

	/**
	 * \brief Creates a program from the given source file
	 *
	 * This Method creates a program containing the source code.
	 * This program is attached to this device and all streams are
	 * executed there.
	 *
	 * \param source The path to the source file
	 *
	 * \see Program
	 */
	hpx::lcos::future<hpx::cuda::program> create_program_with_file(
			std::string file) {
		HPX_ASSERT(this->get_gid());

		std::string source;
		std::string tmp;
		std::ifstream in(file);

		if (!in) {
			std::string errorMessage = "File ";
			errorMessage += file;
			errorMessage += " not found!";
			HPX_THROW_EXCEPTION(hpx::no_success, "create_program_with_file",
					errorMessage);
		}

		while (in) {
			getline(in, tmp);
			source += tmp;
			source += "\n";
		}

		typedef server::device::create_program_with_source_action action_type;
		return hpx::async<action_type>(this->get_gid(), source);
	}

	/**
	 * \brief Creates a buffer attached to this device
	 *
	 * The method creates a buffer with the size specified here and allocates
	 * this on this device.
	 *
	 * \param size The size of the allocated buffer
	 *
	 * \return The buffer with the allocated memoery on this device
	 */
	hpx::lcos::future<hpx::cuda::buffer> create_buffer(size_t size) {
		HPX_ASSERT(this->get_gid());
		typedef server::device::create_buffer_action action_type;
		return hpx::async<action_type>(this->get_gid(), size);
	}

	/**
	 * \brief Synchronous creation of the buffer
	 */
	hpx::cuda::buffer create_buffer_sync(size_t size) {
		HPX_ASSERT(this->get_gid());
		return create_buffer(size).get();
	}
};
}
}
#endif //MANAGED_CUDA_COMPONENT_1_HPP
