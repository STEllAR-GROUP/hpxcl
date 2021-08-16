// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_LCOS_EVENT_HPP_
#define HPX_OPENCL_LCOS_EVENT_HPP_

#include <hpx/hpx.hpp>
#include <hpx/traits/managed_component_policies.hpp>

#include "../export_definitions.hpp"
#include "zerocopy_buffer.hpp"
#include <boost/atomic.hpp>

#include <boost/detail/atomic_count.hpp>

#include <utility>

namespace hpx {
namespace opencl {
namespace lcos {
template <typename Result,
          typename RemoteResult =
              typename traits::promise_remote_result<Result>::type>
class event;
}
}  // namespace opencl
}  // namespace hpx

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
namespace opencl {
namespace lcos {
namespace detail {
///////////////////////////////////////////////////////////////////////////
HPX_OPENCL_EXPORT void unregister_event(hpx::naming::id_type device_id,
                                        hpx::naming::gid_type event_gid);

///////////////////////////////////////////////////////////////////////////
// Zero-copy-Data Function
//

// This function is here for zero-copy of read_to_userbuffer_remote
// Receive a zerocopy_buffer as result of the event.
// De-serialization of the zerocopy_buffer automatically writes
// the data to result_buffer (zerocopy_buffer knows the address of the
// result_buffer's data() member)
// Then set result_buffer as data of this event.
//     template <typename T>
//     void set_zerocopy_data( hpx::naming::id_type event_id,
//                             hpx::opencl::lcos::zerocopy_buffer buf )
//     {
//         typedef hpx::serialization::serialize_buffer<T> buffer_type;
//         typedef hpx::lcos::base_lco_with_value<buffer_type> lco_type;
//         typedef typename
//         hpx::opencl::lcos::event<buffer_type>::shared_state_type
//             event_ptr;
//
//         // Resolve address of lco
//         hpx::naming::address lco_addr = agas::resolve(event_id).get();
//
//         // Ensure everything is correct
//         HPX_ASSERT(hpx::get_locality() == lco_addr.locality_);
//         HPX_ASSERT(traits::component_type_is_compatible<lco_type>::call(lco_addr));
//
//         // Get ptr to lco
//         auto lco_ptr = hpx::get_lva<lco_type>::call(lco_addr.address_);
//
//         // Get ptr to event
//         event_ptr event = static_cast<event_ptr>(lco_ptr);
//
//         // Trigger the event
//         event->set_value(std::move(event->result_buffer));
//     }

//     // Action declaration
//     template <typename T>
//     struct set_zerocopy_data_action
//       : hpx::actions::make_direct_action<
//             void (*)( hpx::naming::id_type,
//             hpx::opencl::lcos::zerocopy_buffer), &set_zerocopy_data<T>,
//             set_zerocopy_data_action<T>
//         >
//     {};

///////////////////////////////////////////////////////////////////////////
template <typename Result, typename RemoteResult>
class event_data : public hpx::lcos::detail::future_data<Result> {
 private:
  typedef hpx::lcos::detail::future_data<Result> parent_type;
  typedef typename parent_type::result_type result_type;

 public:
  typedef typename parent_type::init_no_addref init_no_addref;

  event_data() {}

  event_data(init_no_addref no_addref) : parent_type(no_addref) {}

  ~event_data() {
    HPX_ASSERT(device_id && event_id);
    unregister_event(device_id, event_id.get_gid());
  }

  void init(hpx::naming::id_type&& device_id_) {
    device_id = std::move(device_id_);
  }

  void set_id(hpx::id_type const& id) { event_id = id; }

 public:
  hpx::naming::gid_type get_device_gid() const {
    HPX_ASSERT(device_id);
    return device_id.get_gid();
  }

  hpx::naming::id_type get_event_id() const {
    HPX_ASSERT(event_id);
    return event_id;
  }

 private:
  hpx::naming::id_type device_id;
  hpx::naming::id_type event_id;
};

///////////////////////////////////////////////////////////////////////////
template <>
class event_data<void, hpx::util::unused_type>
    : public hpx::lcos::detail::future_data<void> {
 private:
  typedef hpx::lcos::detail::future_data<void> parent_type;
  typedef parent_type::result_type result_type;

 public:
  typedef typename parent_type::init_no_addref init_no_addref;

  event_data() : is_armed(false) {}

  event_data(init_no_addref no_addref)
      : is_armed(false), parent_type(no_addref) {}

  ~event_data() {
    HPX_ASSERT(device_id && event_id);
    unregister_event(device_id, event_id.get_gid());
  }

  void init(hpx::naming::id_type&& device_id_) {
    device_id = std::move(device_id_);
  }

  void set_id(hpx::id_type const& id) { event_id = id; }

 private:
  boost::atomic<bool> is_armed;

  HPX_OPENCL_EXPORT void arm();

 public:
  // Gets called by when_all, wait_all, etc
  void execute_deferred(error_code& ec = throws) {
    if (!is_armed.exchange(true)) arm();
  }

  // retrieving the value
  result_type* get_result(error_code& ec = throws) {
    this->execute_deferred();
    return this->parent_type::get_result(ec);
  }

  // wait for the value
  void wait(error_code& ec = throws) {
    this->execute_deferred();
    this->parent_type::wait(ec);
  }

  hpx::lcos::future_status wait_until(
      hpx::util::steady_clock::time_point const& abs_time,
      error_code& ec = throws) {
    this->execute_deferred();
    return this->parent_type::wait_until(abs_time, ec);
  }

 public:
  hpx::naming::gid_type get_device_gid() const {
    HPX_ASSERT(device_id);
    return device_id.get_gid();
  }

  hpx::naming::id_type get_event_id() const {
    HPX_ASSERT(event_id);
    return event_id;
  }

 private:
  hpx::naming::id_type device_id;
  hpx::naming::id_type event_id;
};
}  // namespace detail
}  // namespace lcos
}  // namespace opencl
}  // namespace hpx

///////////////////////////////////////////////////////////////////////////////
namespace hpx {
namespace opencl {
namespace lcos {
///////////////////////////////////////////////////////////////////////////
template <typename Result, typename RemoteResult>
class event
    : hpx::lcos::detail::promise_base<
          Result, RemoteResult, detail::event_data<Result, RemoteResult> > {
  typedef hpx::lcos::detail::promise_base<
      Result, RemoteResult, detail::event_data<Result, RemoteResult> >
      base_type;

 public:
  typedef typename base_type::shared_state_type shared_state_type;
  typedef Result result_type;

  /// Construct a new \a event instance. The supplied
  /// \a thread will be notified as soon as the result of the
  /// operation associated with this future instance has been
  /// returned.
  ///
  /// \note         The result of the requested operation is expected to
  ///               be returned as the first parameter using a
  ///               \a base_lco#set_value action. Any error has to be
  ///               reported using a \a base_lco::set_exception action. The
  ///               target for either of these actions has to be this
  ///               future instance (as it has to be sent along
  ///               with the action as the continuation parameter).
  event(hpx::naming::id_type device_id) : base_type() {
    this->shared_state_->init(std::move(device_id));
  }

 public:
  /// Reset the event to allow to restart an asynchronous
  /// operation. Allows any subsequent set_data operation to succeed.
  void reset() {
    this->shared_state_->reset();
    this->future_retrieved_ = false;
  }

  /// \brief Return the global id of this \a future instance
  naming::id_type get_event_id() const {
    return this->shared_state_->get_event_id();
  }

  /// Return whether or not the data is available for this
  /// \a event.
  bool is_ready() const { return this->shared_state_->is_ready(); }

  /// Return whether this instance has been properly initialized
  using base_type::valid;

  using base_type::get_future;
};

///////////////////////////////////////////////////////////////////////////
template <>
class event<void, hpx::util::unused_type>
    : hpx::lcos::detail::promise_base<
          void, hpx::util::unused_type,
          detail::event_data<void, hpx::util::unused_type> > {
  typedef hpx::lcos::detail::promise_base<
      void, hpx::util::unused_type,
      detail::event_data<void, hpx::util::unused_type> >
      base_type;

 public:
  typedef base_type::shared_state_type shared_state_type;
  typedef hpx::util::unused_type result_type;

  /// Construct a new \a event instance. The supplied
  /// \a thread will be notified as soon as the result of the
  /// operation associated with this future instance has been
  /// returned.
  ///
  /// \note         The result of the requested operation is expected to
  ///               be returned as the first parameter using a
  ///               \a base_lco#set_value action. Any error has to be
  ///               reported using a \a base_lco::set_exception action. The
  ///               target for either of these actions has to be this
  ///               future instance (as it has to be sent along
  ///               with the action as the continuation parameter).
  event(hpx::naming::id_type device_id) : base_type() {
    this->shared_state_->init(std::move(device_id));
  }

 public:
  /// Reset the event to allow to restart an asynchronous
  /// operation. Allows any subsequent set_data operation to succeed.
  void reset() {
    this->shared_state_->reset();
    this->future_retrieved_ = false;
  }

  /// \brief Return the global id of this \a future instance
  naming::id_type get_event_id() const {
    return this->shared_state_->get_event_id();
  }

  /// Return whether or not the data is available for this \a event.
  bool is_ready() const { return this->shared_state_->is_ready(); }

  using base_type::get_future;
};
}  // namespace lcos
}  // namespace opencl
}  // namespace hpx

#endif
