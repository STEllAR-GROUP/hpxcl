// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_LCOS_FUTURE_HPP_
#define HPX_OPENCL_LCOS_FUTURE_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include <hpx/lcos/promise.hpp>



///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl { namespace lcos { namespace detail
{

    ///////////////////////////////////////////////////////////////////////////

    void unregister_event( hpx::naming::id_type device_id,
                           boost::uint64_t event_gid_msb,
                           boost::uint64_t event_gid_lsb );

    ///////////////////////////////////////////////////////////////////////////
//    template <typename Future, typename Future::result_type>
    template <typename Future>
    class future_base
    {
        
        public:
            future_base( Future && future_, 
                         hpx::naming::id_type && event_id_ )
                : future    (std::move( future_   )),
                  event_id  (std::move( event_id_ ))
            {
            }
    
            hpx::naming::id_type get_event_id() const
            {
                return event_id;
            }
    
        protected:
            Future future;
            hpx::naming::id_type event_id;

    };


}}}}


///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl { namespace lcos
{

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result>
    class future : public detail::future_base<hpx::future<Result> >
    {
        public:
            future( hpx::future<Result> && future_,
                    hpx::naming::id_type && event_id_ )
                : detail::future_base<hpx::future<Result> >
                    ( std::move(future_), std::move(event_id_) )
            {
            }
    };

    template <typename Result>
    class shared_future : public detail::future_base<hpx::shared_future<Result> >
    {
        public:
            shared_future( hpx::shared_future<Result> && future_,
                           hpx::naming::id_type && event_id_ )
                : detail::future_base<hpx::shared_future<Result> >
                    ( std::move(future_), std::move(event_id_) )
            {
            }
    };


}}};

namespace hpx { namespace opencl { 

    template<typename T>
    using future = lcos::future<T>;

    template<typename T>
    using shared_future = lcos::shared_future<T>;

}};

#endif
