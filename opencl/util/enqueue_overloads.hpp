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

namespace hpx{ namespace opencl{ namespace util{ namespace enqueue_overloads{

    // This is the function that actually extrudes the GID from the futures.
    template<typename Future>
    hpx::naming::id_type
    extrude_id(Future && fut){
        typedef typename std::remove_reference<Future>::type::result_type
            result_type;
        typedef typename hpx::opencl::lcos::event<result_type>::wrapped_type
            event_type;
        
        auto shared_state = hpx::lcos::detail::get_shared_state(fut);
        const event_type* ev
            = dynamic_cast<const event_type*>(shared_state.get());

        hpx::cout << typeid(shared_state).name() << hpx::endl;
        hpx::cout << ev << hpx::endl;
        hpx::cout << ev->get_gid() << hpx::endl;
        
        return hpx::naming::id_type();
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

    // This function object switches its implementation depending on wether
    // the given value is a container or not
    template<bool is_vector>
    struct extrude_all_ids
    {
    };

    template<>
    struct extrude_all_ids<false>
    {
        template<typename T>
        std::vector<hpx::naming::id_type>
        operator()(T && t){
            return std::vector<hpx::naming::id_type>({extrude_id(t)});
        }
    };

    template<>
    struct extrude_all_ids<true>
    {
        template<typename T>
        std::vector<hpx::naming::id_type>
        operator()(const std::vector<T> & t_vec){
            std::vector<hpx::naming::id_type> res;
            res.reserve(t_vec.size());
            for(const T & t : t_vec){
                res.push_back(extrude_id(t));
            }
            return res;
        }
    };


    // The resolver recursive template functions are here to convert
    // an arbitrary number of future and std::vector<future> to
    // one single std::vector<id_type>.
    std::vector<hpx::naming::id_type>
    resolver();

    template<typename Dep>
    std::vector<hpx::naming::id_type>
    resolver(Dep&& dep){
        return extrude_all_ids<detail::is_container<Dep>::value>()(dep);
    }

    template<typename Dep, typename ...Deps>
    std::vector<hpx::naming::id_type>
    resolver(Dep&& dep, Deps&&... deps){

        // recursive call
        std::vector<hpx::naming::id_type> result
            = resolver(std::forward<Deps>(deps)...);

        std::vector<hpx::naming::id_type> ids
            = extrude_all_ids<detail::is_container<Dep>::value>()(dep);

        result.insert(result.end(), ids.begin(), ids.end());
        return result;
    }


}}}}


//TODO avoid id_type copies, move everything (especially inside of resolver)

#define HPX_OPENCL_GENERATE_ENQUEUE_OVERLOADS(return_value, name, args...)      \
                                                                                \
    return_value                                                                \
    name##_impl(args, std::vector<hpx::naming::id_type> &&);                       \
                                                                                \
    /*                                                                          \
     * This class  splits the arguments from the dependencies.                  \
     * It will then create the solution id_type vector via recursive            \
     * templates and call the function name_impl(args, deps).                   \
     */                                                                         \
    template<typename ...Nondeps>                                               \
    class name##_caller{                                                        \
        public:                                                                 \
        template<typename C, typename ...Deps>                                  \
        return_value operator()(C && c, Nondeps &&... nondeps,                  \
                                                     Deps &&... deps)           \
        {                                                                       \
            using hpx::opencl::util::enqueue_overloads::resolver;               \
            return c->name##_impl ( std::forward<Nondeps>(nondeps)...,          \
                    std::move(resolver(std::forward<Deps>(deps)...)) );         \
        }                                                                       \
    };                                                                          \
                                                                                \
    template<typename ...Params>                                                \
    return_value                                                                \
    name (Params &&... params)                                                  \
    {                                                                           \
        return name##_caller<args>()(this, std::forward<Params>(params)...);    \
    }




#endif
