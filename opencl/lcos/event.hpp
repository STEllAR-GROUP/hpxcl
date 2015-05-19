// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_LCOS_EVENT_HPP_
#define HPX_OPENCL_LCOS_EVENT_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include <hpx/lcos/promise.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename RemoteResult>
    class event
      : public hpx::lcos::detail::promise<Result, RemoteResult>
    {

    public:
        
        virtual
        ~event()
        {
            hpx::cout << "event destroyed!" << hpx::endl;

        }


        //////////////////////////////////////////////////////////
        // HPX Stuff
        //
    public:
        enum { value = components::component_promise };

    private:
        template <typename>
        friend struct components::detail_adl_barrier::init;

        void set_back_ptr(components::managed_component<event>* bp)
        {
            HPX_ASSERT(bp);
            HPX_ASSERT(this->gid_ == naming::invalid_gid);
            this->gid_ = bp->get_base_gid();
        }
    };
}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace detail
{
    // use the promise heap factory for constructing events
    template <typename Result, typename RemoteResult>
    struct heap_factory<
            opencl::lcos::detail::event<Result, RemoteResult>,
            managed_component<opencl::lcos::detail::event<Result, RemoteResult> > >
        : promise_heap_factory<opencl::lcos::detail::event<Result, RemoteResult> >
    {};
}}}

namespace hpx { namespace traits
{
    template <typename Result, typename RemoteResult>
    struct managed_component_dtor_policy<
        opencl::lcos::detail::event<Result, RemoteResult> >
    {
        typedef managed_object_is_lifetime_controlled type;
    };
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl { namespace lcos
{
    template <typename Result,
        typename RemoteResult =
            typename traits::promise_remote_result<Result>::type>
    class event;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename RemoteResult>
    class event
    {
    public:
        typedef detail::event<Result, RemoteResult> wrapped_type;
        typedef components::managed_component<wrapped_type> wrapping_type;

    public:
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
            LLCO_(info) << "event::event(" << impl_->get_gid() << ")";
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

        /// Return whether this instance has been properly initialized
        bool valid() const
        {
            return impl_;
        }

        typedef Result result_type;

        virtual ~event()
        {}

        hpx::lcos::future<Result> get_future(error_code& ec = throws)
        {
            if (future_obtained_) {
                HPX_THROWS_IF(ec, future_already_retrieved,
                    "event<Result>::get_future",
                    "future already has been retrieved from this packaged_action");
                return hpx::lcos::future<Result>();
            }

            future_obtained_ = true;

            using traits::future_access;
            return future_access<future<Result> >::create(impl_->get());
        }

        ///
        template <typename T>
        void set_value(T && result)
        {
            (*impl_)->set_data(std::forward<T>(result));
        }

        void set_exception(boost::exception_ptr const& e)
        {
            (*impl_)->set_exception(e);      // set the received error
        }

        template <typename T>
        void set_local_data(T && result)
        {
            (*impl_)->set_local_data(std::forward<Result>(result));
        }

    protected:
        boost::intrusive_ptr<wrapping_type> impl_;
        bool future_obtained_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class event<void, hpx::util::unused_type>
    {
    public:
        typedef detail::event<void, hpx::util::unused_type> wrapped_type;
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

        typedef hpx::util::unused_type result_type;

        ~event()
        {}

        hpx::lcos::future<void> get_future(error_code& ec = throws)
        {
            if (future_obtained_) {
                HPX_THROWS_IF(ec, future_already_retrieved,
                    "event<void>::get_future",
                    "future already has been retrieved from this packaged_action");
                return hpx::lcos::future<void>();
            }

            future_obtained_ = true;

            using traits::future_access;
            return future_access<future<void> >::create(impl_->get());
        }

        void set_value()
        {
            (*impl_)->set_data(hpx::util::unused);
        }

        void set_exception(boost::exception_ptr const& e)
        {
            (*impl_)->set_exception(e);
        }

    protected:
        boost::intrusive_ptr<wrapping_type> impl_;
        bool future_obtained_;
    };
}}};

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    namespace detail
    {
        HPX_EXPORT extern boost::detail::atomic_count unique_type;
    }

    template <typename Result, typename RemoteResult>
    struct component_type_database<
        hpx::opencl::lcos::detail::event<Result, RemoteResult> >
    {
        static components::component_type value;

        static components::component_type get()
        {
            // Events are never created remotely, their factories are not
            // registered with AGAS, so we can assign the component types
            // locally.
            if (value == components::component_invalid)
            {
                value = derived_component_type(++detail::unique_type,
                    components::component_base_lco_with_value);
            }
            return value;
        }

        static void set(components::component_type t)
        {
            HPX_ASSERT(false);
        }
    };

    template <typename Result, typename RemoteResult>
    components::component_type component_type_database<
        hpx::opencl::lcos::detail::event<Result, RemoteResult>
    >::value = components::component_invalid;
}}

#endif
