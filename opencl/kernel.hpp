// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_KERNEL_HPP_
#define HPX_OPENCL_KERNEL_HPP_

// Default includes
#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

// Export definitions
#include "export_definitions.hpp"

// Forward Declarations
#include "fwd_declarations.hpp"

// OpenCL Headers
#include "cl_headers.hpp"

// Crazy function overloading
#include "util/enqueue_overloads.hpp"

namespace hpx {
namespace opencl {

////////////////////////
/// @brief Kernel execution dimensions.
///
/// This structure offers an alternative way to set and reuse kernel
/// execution dimensions.
///
/// Example:
/// \code{.cpp}
///     // Create work_size object
///     hpx::opencl::work_size<1> dim;
///
///     // Set dimensions.
///     dim[0].offset = 0;
///     dim[0].size = 2048;
///
///     // Set local work size.
///     // This can be left out.
///     // OpenCL will then automatically determine the best local work size.
///     dim[0].local_size = 64;
///
///     // Enqueue a kernel using the work_size object
///     event kernel_event = kernel.enqueue(dim).get();
///
/// \endcode
///
template <std::size_t DIM>
struct work_size {
 private:
  struct dimension {
    std::size_t offset;
    std::size_t size;
    std::size_t local_size;
    dimension() {
      offset = 0;
      size = 0;
      local_size = 0;
    }
  };

 private:
  // local_size be treated as NULL if all dimensions have local_size == 0
  dimension dims[DIM];

 public:
  dimension &operator[](std::size_t idx) { return dims[idx]; }
};

//////////////////////////////////////
/// @brief An OpenCL kernel.
///
/// Every kernel belongs to one \ref device.
///
class HPX_OPENCL_EXPORT kernel
    : public hpx::components::client_base<kernel, server::kernel> {
  typedef hpx::components::client_base<kernel, server::kernel> base_type;

 public:
  // Empty constructor, necessary for hpx purposes
  kernel() {}

  // Constructor
  kernel(hpx::shared_future<hpx::naming::id_type> const &gid,
         hpx::naming::id_type device_gid_)
      : base_type(gid), device_gid(std::move(device_gid_)) {}

  kernel(hpx::future<hpx::naming::id_type> &&gid)
      : base_type(std::move(gid)), device_gid() {}

  // initialization

  // ///////////////////////////////////////////////
  // Exposed Component functionality
  //

  /**
   *  @brief Sets a kernel argument
   *
   *  This is the non-blocking version of set_arg
   *
   *  @param arg_index    The argument index to which the buffer will
   *                      be connected.
   *  @param arg          The \ref buffer that will be connected.
   *  @return             A future that will trigger upon completion.
   */
  hpx::lcos::future<void> set_arg_async(cl_uint arg_index,
                                        const hpx::opencl::buffer &arg) const;

  /**
   *  @brief Sets a kernel argument
   *
   *  @param arg_index    The argument index to which the buffer will
   *                      be connected.
   *  @param arg          The \ref buffer that will be connected.
   *  @return             A future that will trigger upon completion.
   */
  void set_arg(cl_uint arg_index, const hpx::opencl::buffer &arg) const;

  /**
   *  @name Starts execution of a kernel, using work_size as work
   *        dimensions.
   *
   *  @param size     The work dimensions on which the kernel should
   *                  get executed on.
   *  @return         An \ref event that triggers upon completion.
   */
  template <std::size_t DIM, typename... Deps>
  hpx::lcos::future<void> enqueue(hpx::opencl::work_size<DIM> size,
                                  Deps &&...dependencies) const;

  hpx::lcos::future<void> enqueue_impl(
      std::vector<std::size_t> &&size_vec,
      hpx::opencl::util::resolved_events &&deps) const;

 protected:
  void ensure_device_id() const;

 private:
  mutable hpx::naming::id_type device_gid;

 private:
  // serialization support
  friend class hpx::serialization::access;

  template <typename Archive>
  void serialize(Archive &ar, unsigned) {
    HPX_ASSERT(device_gid);
    ar &hpx::serialization::base_object<base_type>(*this);
    ar &device_gid;
  }
};

}  // namespace opencl
}  // namespace hpx

////////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATIONS
//
template <std::size_t DIM, typename... Deps>
hpx::future<void> hpx::opencl::kernel::enqueue(hpx::opencl::work_size<DIM> size,
                                               Deps &&...dependencies) const {
  ensure_device_id();

  // combine dependency futures in one std::vector
  using hpx::opencl::util::enqueue_overloads::resolver;
  auto deps =
      resolver(device_gid.get_gid(), std::forward<Deps>(dependencies)...);
  HPX_ASSERT(deps.are_from_device(device_gid));

  // extract information from work_size struct
  std::vector<std::size_t> size_vec(3 * DIM);
  for (std::size_t i = 0; i < DIM; i++) {
    size_vec[i + 0 * DIM] = size[i].offset;
    size_vec[i + 1 * DIM] = size[i].size;
    size_vec[i + 2 * DIM] = size[i].local_size;
  }

  // forward to enqueue_impl
  return enqueue_impl(std::move(size_vec), std::move(deps));
}

#endif
