// Copyright (c)		2013 Damond Howard
//						2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#ifndef HPX_CUDA_PROGRAM_HPP_
#define HPX_CUDA_PROGRAM_HPP_

#include <hpx/include/components.hpp>
#include "server/program.hpp"

namespace hpx {
namespace cuda {

/** \brief This class represents a specific CUDA program containing the CUDA kernel.
 *
 * This class manage the execution of a specific CUDA kernel. It is possible to use
 * pre-compiled CUDA kernels or provide small CUDA kernels and compile them on run time.
 *
 * A program contain one default cudaStream where the kernel are attached to, if there
 * is no stream defined. For multi streaming applications new streams can be created within
 * a program. The kernel related to this program are pinned to this streams.
 *
 */

class program: public hpx::components::client_base<program, server::program> {
	typedef hpx::components::client_base<program, server::program> base_type;

public:

	program() {
	}

	program(hpx::future<hpx::naming::id_type> && gid) :
			base_type(std::move(gid)) {
	}

	/** \brief This method compiles the set source code
	 *
	 * This methods uses the nvrtc library to compile the CUDA source code. You
	 * should use this method only for compiling small CUDA kernels for testing
	 * or for only accelerating small parts of your existing code. For more sophisticated
	 * CUDA application you should use the pre-complied method.
	 *
	 * \param compilerFlags A list with all compiler flags passed to the nvcc compiler
	 * \param debug Compile with debug flags
	 *
	 * \note It is not possible to use include headers, compiling the kernel with nvrtc
	 * \note Compiling a program in debug modus adds -G and -lineinfo to the nvcc compiler
	 * 	flags
	 *
	 */

	hpx::lcos::future<void> build(std::vector<std::string> compilerFlags,
			unsigned int debug = 0) {
		HPX_ASSERT(this->get_gid());
		typedef server::program::build_action action_type;
		return hpx::async<action_type>(this->get_gid(), compilerFlags, debug);
	}

	/**
	 * \brief Synchronous compilation of the source code
	 */

	void build_sync(std::vector<std::string> compilerFlags, unsigned int debug =
			0) {
		// HPX_ASSERT(this->get_gid());
		build(compilerFlags, debug).get();
	}

	/**
	 * \brief Synchronous setting source code
	 */
	void set_source_sync(std::string source) {
		HPX_ASSERT(this->get_gid());
		typedef server::program::set_source_action action_type;
		hpx::async<action_type>(this->get_gid(), source).get();
	}

	/**
	 * \brief This method executes the kernel, compiled or set to this program
	 *
	 * \param modulename The name of the kernel
	 * \param args The function arguments passed to the kernel
	 * \param grid The dimensions of the grid size
	 * \param block The dimensions of the block size
	 * \param stream The stream at which the kernel is attached to
	 *
	 * \note Each program has a default stream, which is not the same as the default stream
	 * 	of the CUDA API. Not setting the last parameter implies that the kernel is executed
	 * 	on the default stream of this program.
	 */

	hpx::lcos::future<void> run(std::vector<hpx::cuda::buffer> args,
			std::string modulename, hpx::cuda::server::program::Dim3 grid,
			hpx::cuda::server::program::Dim3 block, unsigned int stream = 0) {
		HPX_ASSERT(this->get_gid());
		std::vector<intptr_t> args_pointer;

		typedef server::program::run_action action_type;
		return hpx::async<action_type>(this->get_gid(), args_pointer,
				modulename, grid, block, stream);

	}
};
}
}
#endif //PROGRAM_1_HPP
