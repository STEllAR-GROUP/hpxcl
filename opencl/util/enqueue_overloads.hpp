// Copyright (c)    2015 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_UTIL_ENQUEUE_OVERLOADS_HPP_
#define HPX_OPENCL_UTIL_ENQUEUE_OVERLOADS_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

namespace hpx{ namespace opencl{ namespace util{ namespace dependencies{

    // This is the function that actually extrudes the GID from the futures.
    template<typename future_type>
    hpx::naming::id_type
    extrude_id(future_type && fut){
        return hpx::naming::id_type();
    }

    // The resolver recursive template functions are here to convert
    // an arbitrary number of future and std::vector<future> to
    // one single std::vector<id_type>.
    template<typename Dep>
    std::vector<hpx::naming::id_type>
    resolver(Dep&& dep){
        std::vector<hpx::naming::id_type> result;
        result.push_back(extrude_id(std::forward<Dep>(dep)));
        return result;
    }

    template<typename Dep>
    std::vector<hpx::naming::id_type>
    resolver(std::vector<Dep> & dep_vec){
        std::vector<hpx::naming::id_type> result;
        for(const Dep & dep : dep_vec){
            result.push_back(extrude_id(dep));
        }
        return result;
    }

    template<typename Dep, typename ...Deps>
    std::vector<hpx::naming::id_type>
    resolver(Dep&& dep, Deps&&... deps){
        std::vector<hpx::naming::id_type> result
            = resolver(std::forward<Deps>(deps)...);
        result.push_back(extrude_id(dep));
        return result;
    }

    template<typename Dep, typename ...Deps>
    std::vector<hpx::naming::id_type>
    resolver(std::vector<Dep> & dep_vec, Deps&&... deps){
        std::vector<hpx::naming::id_type> result
            = resolver(std::forward<Deps>(deps)...);
        for(const Dep & dep : dep_vec){
            result.push_back(extrude_id(dep));
        }
        return result;
    }

    // This class is there to split the arguments from the dependencies.
    // It will then create the solution id_type vector via recursive
    // templates and call the function name_impl(args, deps).
    template<typename return_value, typename ...Args>
    class caller{
        public:
        
        template<typename F, typename ...Deps>
        hpx::future<return_value> operator()(F && f, Args &&... args,
                                                     Deps &&... deps)
        {
            return f( std::forward<Args>(args)...,
                      resolver(std::forward<Deps>(deps)...) );
        }
    };

}}}}

#define HPX_OPENCL_GENERATE_ENQUEUE_OVERLOADS(return_value, name, args...)      \
                                                                                \
    static hpx::future<return_value>                                                   \
    name##_impl(args, std::vector<hpx::naming::id_type>);                       \
                                                                                \
    class name##_type{                                                          \
        public:                                                                 \
            template<typename ...Params>                                        \
            hpx::future<return_value> operator()(Params &&... params){          \
                using hpx::opencl::util::dependencies::caller;                  \
                return caller<return_value, args>()(name##_impl,                \
                            std::forward<Params>(params)...);                   \
            }                                                                   \
    } name



#endif
