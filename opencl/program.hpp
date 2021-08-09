// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_PROGRAM_HPP_
#define HPX_OPENCL_PROGRAM_HPP_

// Default includes
#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

// Export definitions
#include "export_definitions.hpp"

// Forward Declarations
#include "fwd_declarations.hpp"

namespace hpx {
namespace opencl {

//////////////////////////////////////
/// @brief An OpenCL program consisting of one or multiple kernels.
///
/// Every program belongs to one \ref device.
///
class HPX_OPENCL_EXPORT program
    : public hpx::components::client_base<program, server::program> {
  typedef hpx::components::client_base<program, server::program> base_type;

 public:
  // Empty constructor, necessary for hpx purposes
  program() {}

  // Constructor
  program(hpx::shared_future<hpx::naming::id_type> const& gid,
          hpx::naming::id_type device_gid_)
      : base_type(gid), device_gid(std::move(device_gid_)) {}

  program(hpx::future<hpx::naming::id_type>&& gid)
      : base_type(std::move(gid)), device_gid() {}

  // initialization

  // ///////////////////////////////////////////////
  // Exposed Component functionality
  //

  /**
   *  @brief Builds the program, non-blocking.
   *
   *  @return A future that will trigger upon build completion.
   */
  hpx::lcos::future<void> build_async() const;

  /**
   *  @brief Builds the program, non-blocking.
   *
   *  @param build_options    A string with specific build options.<BR>
   *                          Look at the official
   * <A
   * HREF="http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clBuildProgram.html">OpenCL
   * Reference</A> for further information.
   *  @return A future that will trigger upon build completion.
   */
  hpx::lcos::future<void> build_async(std::string build_options) const;

  /**
   *  @brief Builds the program, blocking.
   *
   *  @return A future that will trigger upon build completion.
   */
  void build() const;

  /**
   *  @brief Builds the program, blocking.
   *
   *  @param build_options    A string with specific build options.<BR>
   *                          Look at the official
   * <A
   * HREF="http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clBuildProgram.html">OpenCL
   * Reference</A> for further information.
   *  @return A future that will trigger upon build completion.
   */
  void build(std::string build_options) const;

  /**
   *  @brief Retrieves the binary of a built program.
   *         It can be used to create programs with
   *         device::create_program_with_binary().
   *
   *  @return A future to the binary code
   */
  hpx::lcos::future<hpx::serialization::serialize_buffer<char> > get_binary()
      const;

  /**
   *  @brief Creates a kernel.
   *
   *  The kernel with the name kernel_name has to be defined and
   *  implemented in the program source code.
   *
   *  @param kernel_name  The name of the kernel to be created
   *  @return             A kernel object.
   */
  hpx::opencl::kernel create_kernel(std::string kernel_name) const;

 protected:
  void ensure_device_id() const;

 private:
  mutable hpx::naming::id_type device_gid;

 private:
  // serialization support
  friend class hpx::serialization::access;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    HPX_ASSERT(device_gid);
    ar& hpx::serialization::base_object<base_type>(*this);
    ar& device_gid;
  }
};

}  // namespace opencl
}  // namespace hpx

#endif
