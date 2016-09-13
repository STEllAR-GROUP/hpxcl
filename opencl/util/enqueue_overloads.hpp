// Copyright (c)    2015 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_UTIL_ENQUEUE_OVERLOADS_HPP_
#define HPX_OPENCL_UTIL_ENQUEUE_OVERLOADS_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include <hpx/include/iostreams.hpp>

#include "../lcos/event.hpp"

namespace hpx { namespace opencl { namespace util
{
    struct resolved_events
    {
    public:
        std::vector<hpx::naming::id_type> event_ids;
        std::vector<hpx::naming::gid_type> device_ids;
        bool are_from_device(const hpx::naming::id_type& device_id)
        {
            hpx::naming::gid_type device_gid = device_id.get_gid();
            for(const auto& id : device_ids){
                if(device_gid != id)
                    return false;
            }
            return true;
        }
        bool are_from_devices( const hpx::naming::id_type& device1,
                                const hpx::naming::id_type& device2 )
        {
            hpx::naming::gid_type device_gid1 = device1.get_gid();
            hpx::naming::gid_type device_gid2 = device2.get_gid();
            for(const auto& id : device_ids){
                if((device_gid1 != id) && (device_gid2 != id))
                    return false;
            }
            return true;
        }
    };
}}}

namespace hpx { namespace opencl { namespace util { namespace enqueue_overloads
{
    // TODO implement check for correct device
    // This is the function that actually extrudes the GID from the futures.
    template<typename Future>
    hpx::naming::id_type
    extrude_id(const Future & fut, hpx::naming::gid_type& device_id)
    {
        typedef typename std::remove_reference<Future>::type::result_type
            result_type;
        typedef typename hpx::opencl::lcos::event<result_type>::shared_state_type
            event_type;

        auto shared_state = hpx::traits::detail::get_shared_state(fut);

        HPX_ASSERT(boost::dynamic_pointer_cast<event_type>(shared_state).get());
        auto ev = boost::static_pointer_cast<event_type>(shared_state);

        HPX_ASSERT_MSG(device_id != ev->get_device_gid(), (std::stringstream() << "devide_id=" << device_id << " and ev_id=" << ev->get_device_id()));

        auto event_id = ev->get_event_id();
        return event_id;
    }

    namespace detail
    {
        BOOST_MPL_HAS_XXX_TRAIT_DEF(value_type)
        BOOST_MPL_HAS_XXX_TRAIT_DEF(iterator)
        BOOST_MPL_HAS_XXX_TRAIT_DEF(size_type)
        BOOST_MPL_HAS_XXX_TRAIT_DEF(reference)

        template <typename T>
        struct is_container
          : boost::mpl::bool_<
                has_value_type<T>::value && has_iterator<T>::value &&
                has_size_type<T>::value && has_reference<T>::value>
        {};

        template <typename T>
        struct is_container<T&>
          : is_container<T>
        {};
    }

    // This function object switches its implementation depending on whether
    // the given value is a container or not
    template<bool is_vector>
    struct extrude_all_ids
    {
    };

    template<>
    struct extrude_all_ids<false>
    {
        template<typename T>
        void
        operator()(const T & t,
                   std::vector<hpx::naming::id_type> &event_ids,
                   std::vector<hpx::naming::gid_type> &device_ids) const
        {
            hpx::naming::gid_type device_id;
            event_ids.push_back(std::move(extrude_id(t, device_id)));
            device_ids.push_back(std::move(device_id));
        }
    };

    template<>
    struct extrude_all_ids<true>
    {
        template<typename T>
        void
        operator()(const std::vector<T> & t_vec,
                   std::vector<hpx::naming::id_type> &event_ids,
                   std::vector<hpx::naming::gid_type> &device_ids) const
        {
            for(const T & t : t_vec){
                hpx::naming::gid_type device_id;
                event_ids.push_back(std::move(extrude_id(t, device_id)));
                device_ids.push_back(std::move(device_id));
            }
        }
    };


    // The resolver recursive template functions are here to convert
    // an arbitrary number of future and std::vector<future> to
    // one single std::vector<id_type>.
    HPX_OPENCL_EXPORT void
    resolver_impl(std::vector<hpx::naming::id_type>&,
                  std::vector<hpx::naming::gid_type>&);

    template<typename Dep>
    void
    resolver_impl(std::vector<hpx::naming::id_type>& event_ids,
                  std::vector<hpx::naming::gid_type>& device_ids,
                  Dep&& dep)
    {
        extrude_all_ids<detail::is_container<Dep>::value>()( dep, event_ids,
                                                                  device_ids);
    }

    template<typename Dep, typename ...Deps>
    void
    resolver_impl(std::vector<hpx::naming::id_type>& event_ids,
                  std::vector<hpx::naming::gid_type>& device_ids,
                  Dep&& dep, Deps&&... deps)
    {
        // process current dep
        extrude_all_ids<detail::is_container<Dep>::value>()( dep, event_ids,
                                                                  device_ids );

        // recursive call
        resolver_impl(event_ids, device_ids, std::forward<Deps>(deps)...);
    }

    template<typename ...Deps>
    resolved_events
    resolver(Deps&&... deps)
    {
        resolved_events res;
        res.event_ids.reserve(sizeof...(deps));
        res.device_ids.reserve(sizeof...(deps));
        resolver_impl( res.event_ids, res.device_ids,
                       std::forward<Deps>(deps)... );
        return res;
    }

}}}}

// #define HPX_OPENCL_GENERATE_ENQUEUE_OVERLOADS(return_value, name, ...)          \
//                                                                                 \
//     return_value                                                                \
//     name##_impl(__VA_ARGS__, hpx::opencl::util::resolved_events &&);            \
//                                                                                 \
//     /*                                                                          \
//      * This class  splits the arguments from the dependencies.                  \
//      * It will then create the solution id_type vector via recursive            \
//      * templates and call the function name_impl(args, deps).                   \
//      */                                                                         \
//     template<typename ...Nondeps>                                               \
//     class name##_caller{                                                        \
//         public:                                                                 \
//         template<typename C, typename ...Deps>                                  \
//         return_value operator()(C && c, Nondeps &&... nondeps,                  \
//                                                      Deps &&... deps)           \
//         {                                                                       \
//             using hpx::opencl::util::enqueue_overloads::resolver;               \
//             return c->name##_impl ( std::forward<Nondeps>(nondeps)...,          \
//                     std::move(resolver(std::forward<Deps>(deps)...)) );         \
//         }                                                                       \
//     };                                                                          \
//                                                                                 \
//     template<typename ...Params>                                                \
//     return_value                                                                \
//     name (Params &&... params)                                                  \
//     {                                                                           \
//         return name##_caller<__VA_ARGS__>()(this, std::forward<Params>(params)...); \
//     }

#endif
