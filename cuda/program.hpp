// Copyright (c)		2013 Damond Howard
//						2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt
#pragma once
#ifndef HPX_CUDA_PROGRAM_HPP_
#define HPX_CUDA_PROGRAM_HPP_

#include <hpx/include/serialization.hpp>

#include <hpx/include/components.hpp>
#include "cuda/server/program.hpp"
#include "cuda/buffer.hpp"

namespace hpx {
namespace cuda {

/** \brief This class represents a specific CUDA program containing the
 * CUDA kernel.
 *
 * This class manage the execution of a specific CUDA kernel. The CUDA kernel
 * is compiled on run time with the nvrtc library.
 *
 * A program contain one default cudaStream where the kernel are attached to,
 * if there is no stream defined. For multi-streaming applications new
 * streams can be created within a program. The kernel related to
 * this program are pinned to this streams.
 *
 */
class program : public hpx::components::client_base<program, server::program> {
  typedef hpx::components::client_base<program, server::program> base_type;

 public:
  program() {}

  program(hpx::future<hpx::naming::id_type>&& gid)
      : base_type(std::move(gid)) {}

  /** \brief This method compiles the set source code
   *
   * This methods uses the nvrtc library to compile the CUDA source code. You
   * should use this method only for compiling small CUDA kernels for testing
   * or for only accelerating small parts of your existing code.
   *
   * \param compilerFlags A list with all compiler flags passed to
   *	the nvcc compiler
   * \param modulename Name of the kernel, which to be compiled
   * \param debug Compile with debug flags
   *
   * \note It is not possible to use included headers.
   * \note Compiling a program in debug modus adds -G and
   * 	-lineinfo to the nvcc compiler flags
   *
   */

  hpx::lcos::future<void> build(std::vector<std::string> compilerFlags,
                                std::vector<std::string> modulenames,
                                unsigned int debug = 0) {
    HPX_ASSERT(this->get_id());
    typedef server::program::build_action action_type;
    return hpx::async<action_type>(this->get_id(), compilerFlags, modulenames,
                                   debug);
  }

  /**
   * \brief Synchronous compilation of the source code
   */

  void build_sync(std::vector<std::string> compilerFlags,
                  std::string modulename, unsigned int debug = 0) {
    std::vector<std::string> modulenames;
    modulenames.push_back(modulename);
    build(compilerFlags, modulenames, debug).get();
  }

  hpx::lcos::future<void> build(std::vector<std::string> compilerFlags,
                                std::string modulename,
                                unsigned int debug = 0) {
    HPX_ASSERT(this->get_id());
    std::vector<std::string> modulenames;
    modulenames.push_back(modulename);
    typedef server::program::build_action action_type;
    return hpx::async<action_type>(this->get_id(), compilerFlags, modulenames,
                                   debug);
  }

  /**
   * \brief Synchronous compilation of the source code
   */

  void build_sync(std::vector<std::string> compilerFlags,
                  std::vector<std::string> modulenames,
                  unsigned int debug = 0) {
    build(compilerFlags, modulenames, debug).get();
  }

  /**
   * \brief Synchronous setting source code
   */
  void set_source_sync(std::string source) {
    HPX_ASSERT(this->get_id());
    typedef server::program::set_source_action action_type;
    hpx::async<action_type>(this->get_id(), source).get();
  }

#ifdef HPXCL_CUDA_WITH_STREAMS
  /**
   * \brief This method executes the kernel, compiled or set to this program
   *
   * \param modulename The name of the kernel
   * \param args The function arguments passed to the kernel
   * \param grid The dimensions of the grid size
   * \param block The dimensions of the block size
   * \param stream The stream at which the kernel is attached to
   *
   * \note Each program has a default stream, which is not the same as the
   * 	default stream of the CUDA API. Not setting the last parameter
   * 	implies that the kernel is executed on the default stream
   * 	of this program.
   */

  hpx::lcos::future<void> run(std::vector<hpx::cuda::buffer> args,
                              std::string modulename,
                              hpx::cuda::server::program::Dim3 grid,
                              hpx::cuda::server::program::Dim3 block,
                              size_t shared_memory, int stream = -1) {
    HPX_ASSERT(this->get_id());

    std::vector<hpx::naming::id_type> args_id;

    for (unsigned int i = 0; i < args.size(); i++) {
      args_id.push_back(args[i].get_id());
    }

    std::vector<hpx::naming::id_type> dependencies;

    typedef server::program::run_action action_type;
    return hpx::async<action_type>(this->get_id(), args_id, modulename, grid,
                                   block, dependencies, shared_memory, stream);
  }

  /**
   * \brief This method executes the kernel, compiled or set to this program
   *
   * \param modulename The name of the kernel
   * \param args The function arguments passed to the kernel
   * \param grid The dimensions of the grid size
   * \param block The dimensions of the block size
   * \param dependencies The data, the kernel execution depends on
   * \param stream The stream at which the kernel is attached to
   *
   * \note Each program has a default stream, which is not the same
   * 	as the default stream of the CUDA API. Not setting the last parameter
   * 	implies that the kernel is executed on the default stream.
   */

  hpx::lcos::future<void> run(std::vector<hpx::cuda::buffer> args,
                              std::string modulename,
                              hpx::cuda::server::program::Dim3 grid,
                              hpx::cuda::server::program::Dim3 block,
                              std::vector<hpx::cuda::buffer> dependencies,
                              size_t shared_memory, int stream = -1) {
    HPX_ASSERT(this->get_id());

    std::vector<hpx::naming::id_type> args_id;

    for (unsigned int i = 0; i < args.size(); i++) {
      args_id.push_back(args[i].get_id());
    }

    std::vector<hpx::naming::id_type> dependencies_id;

    for (unsigned int i = 0; i < dependencies.size(); i++) {
      dependencies_id.push_back(dependencies[i].get_id());
    }

    typedef server::program::run_action action_type;
    return hpx::async<action_type>(this->get_id(), args_id, modulename, grid,
                                   block, dependencies_id, shared_memory,
                                   stream);
  }

  hpx::lcos::future<void> run(std::vector<hpx::cuda::buffer> args,
                              std::string modulename,
                              hpx::cuda::server::program::Dim3 grid,
                              hpx::cuda::server::program::Dim3 block,
                              hpx::cuda::buffer dependency,
                              size_t shared_memory, int stream = -1) {
    HPX_ASSERT(this->get_id());

    std::vector<hpx::naming::id_type> args_id;

    for (unsigned int i = 0; i < args.size(); i++) {
      args_id.push_back(args[i].get_id());
    }

    std::vector<hpx::naming::id_type> dependencies_id;
    dependencies_id.push_back(dependency.get_id());

    typedef server::program::run_action action_type;
    return hpx::async<action_type>(this->get_id(), args_id, modulename, grid,
                                   block, dependencies_id, shared_memory,
                                   stream);
  }

  /**
   * \brief This method returns the number of streams at this device
   *
   * \return Future containing the number of streams created on the device
   *
   */

  hpx::lcos::future<unsigned int> get_streams_size() {
    HPX_ASSERT(this->get_id());
    typedef server::program::get_streams_size_action action_type;
    return hpx::async<action_type>(this->get_id());
  }

  /**
   * \brief This method returns the id of this created stream at the device
   *
   * \return Future containing the streams id
   *
   */

  hpx::lcos::future<unsigned int> create_stream() {
    HPX_ASSERT(this->get_id());
    typedef server::program::create_stream_action action_type;
    return hpx::async<action_type>(this->get_id());
  }
#else
  /**
   *
   *\brief This method executes the kernel set to the program
   *
   *\param args List of arguments to the kernel
   *\param modulename Name of the kernel program to be executed
   *\param grid Grid dimension of the kernel
   *\param block Block dimension of the kernel
   *
   *\note The kernel is executed the null stream provided by CUDA API
   */
  hpx::lcos::future<void> run(std::vector<hpx::cuda::buffer> args,
                              std::string modulename,
                              hpx::cuda::server::program::Dim3 grid,
                              hpx::cuda::server::program::Dim3 block) {
    HPX_ASSERT(this->get_id());

    std::vector<hpx::naming::id_type> args_id;

    for (unsigned int i = 0; i < args.size(); i++) {
      args_id.push_back(args[i].get_id());
    }

    typedef server::program::run_action action_type;
    return hpx::async<action_type>(this->get_id(), args_id, modulename, grid,
                                   block);
  }
#endif
};
}  // namespace cuda
}  // namespace hpx
#endif  // PROGRAM_1_HPP
