// Copyright (c)       2013 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_CUDA_LCOS_EVENT_HPP_
#define HPX_CUDA_LCOS_EVENT_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include <hpx/lcos/promise.hpp>



#include "cuda/fwd_declarations.hpp"
#include "cuda/export_definitions.hpp"

//#include "../server/device_server.cpp"

#include "zerocopy_buffer.hpp"



namespace hpx { namespace cuda { namespace lcos
{
    template <typename Result,
        typename RemoteResult =
            typename traits::promise_remote_result<Result>::type>
    class event;
}}}

namespace hpx { namespace cuda { namespace lcos { namespace detail
{
    template <typename Result, typename RemoteResult>
    class event;
}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace detail
{
    // use the promise heap factory for constructing events
    template <typename Result, typename RemoteResult>
    struct heap_factory<
            cuda::lcos::detail::event<Result, RemoteResult>,
            managed_component<cuda::lcos::detail::event<Result, RemoteResult> > >
        : promise_heap_factory<cuda::lcos::detail::event<Result, RemoteResult> >
    {};
}}}


///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace cuda { namespace lcos { namespace detail
{

    ///////////////////////////////////////////////////////////////////////////

    HPX_CUDA_EXPORT void unregister_event( hpx::naming::id_type device_id,
                           hpx::naming::gid_type event_gid );

    //////////////////////////////////////////////////////////////////////////
    // Zerocopy-Data Function
    //

    // This function is here for zero-copy of read_to_userbuffer_remote
    // Receive a zerocopy_buffer as result of the event.
    // De-serialization of the zerocopy_buffer automatically writes
    // the data to result_buffer (zerocopy_buffer knows the address of the
    // result_buffer's data() member)
    // Then set result_buffer as data of this event.
    template<typename T>
    void set_zerocopy_data( hpx::naming::id_type event_id,
                            hpx::cuda::lcos::zerocopy_buffer buf ){

        typedef hpx::serialization::serialize_buffer<T> buffer_type;
        typedef hpx::lcos::base_lco_with_value<buffer_type> lco_type;
        typedef typename hpx::cuda::lcos::event<buffer_type>::wrapped_type*
            event_ptr;

        // Resolve address of lco
        hpx::naming::address lco_addr = agas::resolve(event_id).get();

        // Ensure everything is correct
        HPX_ASSERT(hpx::get_locality() == lco_addr.locality_);
        HPX_ASSERT(traits::component_type_is_compatible<lco_type>::call(
                        lco_addr));

        // Get ptr to lco
        auto lco_ptr = hpx::get_lva<lco_type>::call(lco_addr.address_);

        // Get ptr to event
        event_ptr event = static_cast<event_ptr>(lco_ptr);

        // Make sure sizes match
        //HPX_ASSERT( buf.size() == event->result_buffer.size() * sizeof(T) );

        // Trigger the event
        event->set_value(std::move(event->result_buffer));
    }

    // Action delcaration
    template <typename T>
    struct set_zerocopy_data_action
      : hpx::actions::make_direct_action<void (*)( hpx::naming::id_type,
                                            hpx::cuda::lcos::zerocopy_buffer ),
            &set_zerocopy_data<T>,
            set_zerocopy_data_action<T> >
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename RemoteResult>
    class event
      : public hpx::lcos::detail::promise<Result, RemoteResult>
    {

    private:
        typedef hpx::lcos::detail::promise<Result, RemoteResult> parent_type;
        typedef typename hpx::lcos::detail::future_data<Result>::result_type
            result_type;

    public:

        event(hpx::naming::id_type && device_id_, Result && result_buffer_)
            : device_id(std::move(device_id_)),
              result_buffer(std::move(result_buffer_))
        {
        }

        virtual
        ~event()
        {
            unregister_event( device_id,
                              this->get_base_gid() );
        }

        ////////////////////////////////////////////////////////////////////////
        // Stuff that enable zero-copy
        //
    public:

        // This holds the buffer that will get returned by the future.
        Result result_buffer;


        ////////////////////////////////////////////////////////////////////////
        // Internal stuff
        //
    public:
        hpx::naming::gid_type get_device_gid()
        {
            return device_id.get_gid();
        }

    private:

        hpx::naming::id_type device_id;


        ////////////////////////////////////////////////////////////////////////
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

        ////////////////////////////////////////////////////////////////////////
        // Stuff that enables id_type lookup
        //
    public:
        hpx::naming::id_type get_event_id(){
            return event_id;
        }

        void retrieve_event_id(){
            event_id = this->get_id();
        }
    private:

        hpx::naming::id_type event_id;

        bool requires_delete()
        {
            boost::unique_lock<naming::gid_type> l(this->gid_.get_mutex());
            long counter = --this->count_;

            // special case: counter == 1. (meaning: agas is the only one
            // still holding a reference to this object. especially,
            // all futures are out of scope.)
            if (1 == counter)
            {
                HPX_ASSERT(event_id.get_gid() != naming::invalid_gid);

                l.unlock();

                // delete local reference to prevent recursive dependency
                event_id = hpx::naming::id_type();

                return false;
            }
            else if (0 == counter)
            {
                return true;
            }
            return false;
        }

    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class event<void, hpx::util::unused_type>
      : public hpx::lcos::detail::promise<void, hpx::util::unused_type>
    {

    private:
        typedef hpx::lcos::detail::promise<void, hpx::util::unused_type>
            parent_type;
        typedef hpx::lcos::detail::future_data<void>::result_type
            result_type;

    public:

        event(hpx::naming::id_type && device_id_)
            : device_id(std::move(device_id_)), is_armed(false)
        {
        }

        virtual
        ~event()
        {
            unregister_event( device_id,
                              this->get_base_gid() );
        }

        ////////////////////////////////////////////////////////////////////////
        // Overrides that enable the event to be deferred
        //
    private:
        boost::atomic<bool> is_armed;

        HPX_CUDA_EXPORT void arm();

    public:
        // Gets called by when_all, wait_all, etc
        virtual void execute_deferred(error_code& ec = throws){
            if(!is_armed.exchange(true)){
                this->arm();
            }
        }

        // retrieving the value
        virtual result_type* get_result(error_code& ec = throws)
        {
            this->execute_deferred();
            return this->parent_type::get_result(ec);
        }

        // wait for the value
        virtual void wait(error_code& ec = throws)
        {
            this->execute_deferred();
            this->parent_type::wait(ec);
        }

        virtual BOOST_SCOPED_ENUM(hpx::lcos::future_status)
        wait_until(boost::chrono::steady_clock::time_point const& abs_time,
            error_code& ec = throws)
        {
            if (!is_armed.load())
                return hpx::lcos::future_status::deferred; //-V110
            else
                return this->parent_type::wait_until(abs_time, ec);
        };

        ////////////////////////////////////////////////////////////////////////
        // Internal stuff
        //
    public:
        hpx::naming::gid_type get_device_gid()
        {
            return device_id.get_gid();
        }

    private:

        hpx::naming::id_type device_id;


        ////////////////////////////////////////////////////////////////////////
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

        ////////////////////////////////////////////////////////////////////////
        // Stuff that enables id_type lookup
        //
    public:
        hpx::naming::id_type get_event_id(){
            return event_id;
        }

        void retrieve_event_id(){
            event_id = get_id();
        }
    private:

        hpx::naming::id_type event_id;

        bool requires_delete()
        {
            boost::unique_lock<naming::gid_type> l(this->gid_.get_mutex());
            long counter = --this->count_;

            // special case: counter == 1. (meaning: agas is the only one
            // still holding a reference to this object. especially,
            // all futures are out of scope.)
            if (1 == counter)
            {
                HPX_ASSERT(event_id.get_gid() != naming::invalid_gid);

                l.unlock();

                // delete local reference to prevent recursive dependency
                event_id = hpx::naming::id_type();

                return false;
            }
            else if (0 == counter)
            {
                return true;
            }
            return false;
        }

    };

}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    template <typename Result, typename RemoteResult>
    struct managed_component_dtor_policy<
        cuda::lcos::detail::event<Result, RemoteResult> >
    {
        typedef managed_object_is_lifetime_controlled type;
    };
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace cuda { namespace lcos
{
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
        event(hpx::naming::id_type device_id, Result result_buffer = Result())
          : impl_(new wrapping_type(
                new wrapped_type(std::move(device_id), std::move(result_buffer))
            )),
            future_obtained_(false)
        {
            LLCO_(info) << "event::event(" << (*impl_)->get_unmanaged_id() << ")";
            (*impl_)->retrieve_event_id();
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
        naming::id_type get_event_id() const
        {
            return (*impl_)->get_event_id();
        }

        /// \brief Return the global id of this \a future instance
        naming::id_type get_id() const
        {
            HPX_ASSERT(false);
            return (*impl_)->get_id();
        }

        /// \brief Return the global id of this \a future instance
        naming::gid_type get_unmanaged_gid() const
        {
            return (*impl_)->get_unmanaged_gid();
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
        event(hpx::naming::id_type device_id)
          : impl_(new wrapping_type( // event<void> is deferred
                new wrapped_type(std::move(device_id))
            )),
            future_obtained_(false)
        {
            LLCO_(info) << "event<void>::event(" << (*impl_)->get_unmanaged_id() << ")";
            (*impl_)->retrieve_event_id();
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
        naming::id_type get_event_id() const
        {
            return (*impl_)->get_event_id();
        }

        /// \brief Return the global id of this \a future instance
        naming::id_type get_id() const
        {
            HPX_ASSERT(false);
            return (*impl_)->get_id();
        }

        /// \brief Return the global id of this \a future instance
        naming::gid_type get_unmanaged_gid() const
        {
            return (*impl_)->get_unmanaged_id().get_gid();
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
        hpx::cuda::lcos::detail::event<Result, RemoteResult> >
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
        hpx::cuda::lcos::detail::event<Result, RemoteResult>
    >::value = components::component_invalid;
}}

#endif
