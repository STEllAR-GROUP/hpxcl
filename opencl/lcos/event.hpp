// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_LCOS_EVENT_HPP_
#define HPX_OPENCL_LCOS_EVENT_HPP_

#include <hpx/config.hpp>
#include <hpx/hpx.hpp>

#include <hpx/lcos/promise.hpp>


///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename RemoteResult>
    class event
      : public hpx::lcos::detail::promise<Result, RemoteResult>
    {
        
    }
}}};

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl { namespace lcos {
{

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class event<void, util::unused_type>
    {
    public:
        typedef detail::event<void, util::unused_type> wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;

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
        event()
          : impl_(new wrapping_type(new wrapped_type())),
            future_obtained_(false)
        {
            LLCO_(info) << "event<void>::event(" << impl_->get_gid() << ")";
        }

    protected:
        template <typename Impl>
        event(Impl* impl)
          : impl_(impl), future_obtained_(false)
        {}

    public:
        /// Reset the event to allow to restart an asynchronous
        /// operation. Allows any subsequent set_data operation to succeed.
        void reset()
        {
            (*impl_)->reset();
            future_obtained_ = false;
        }

        /// \brief Return the global id of this \a future instance
        naming::id_type get_gid() const
        {
            return (*impl_)->get_gid();
        }

        /// \brief Return the global id of this \a future instance
        naming::gid_type get_base_gid() const
        {
            return (*impl_)->get_base_gid();
        }

        /// Return whether or not the data is available for this
        /// \a event.
        bool is_ready() const
        {
            return (*impl_)->is_ready();
        }

        typedef util::unused_type result_type;

        ~event()
        {}

        lcos::future<void> get_future(error_code& ec = throws)
        {
            if (future_obtained_) {
                HPX_THROWS_IF(ec, future_already_retrieved,
                    "event<void>::get_future",
                    "future already has been retrieved from this packaged_action");
                return lcos::future<void>();
            }

            future_obtained_ = true;

            using traits::future_access;
            return future_access<future<void> >::create(impl_->get());
        }

        void set_value()
        {
            (*impl_)->set_data(util::unused);
        }

        void set_exception(boost::exception_ptr const& e)
        {
            (*impl_)->set_exception(e);
        }

    protected:
        boost::intrusive_ptr<wrapping_type> impl_;
        bool future_obtained_;
    };
}
#endif
