// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_ENQUEUE_OVERLOADS_HPP__
#define HPX_OPENCL_ENQUEUE_OVERLOADS_HPP__

#include <hpx/lcos/when_all.hpp>


/*
 * The purpose of this file is to define the overloads of all the functions
 * that follow the common theme:
 *
 *
 *          hpx::lcos::future<event>
 *          class::function([args...], std::vector<event>);
 *
 *      (where function usually starts with enqueue_)
 *
 * following line:
 *
 *          OVERLOAD_FUNCTION(<class>, <function>, <args_with_types>,
 *                                                <args_without_types>);
 *
 * implements the following overloads:
 *          
 *          hpx::lcos::future<event>
 *          class::function([args...]);
 *
 *          hpx::lcos::future<event>
 *          class::function([args...], event);
 *
 *          hpx::lcos::future<event>
 *          class::function([args...], hpx::lcos::shared_future<event>);
 *
 *          hpx::lcos::future<event>
 *          class::function([args...],
 *                          std::vector<hpx::lcos::shared_future<event>>);
 *
 *
 */





// //////////////////////////////////////////////////////////////////////////
//  Function overload defines, that take away a LOT of work.
//  Without these defines, you would have to overload every single function
//  manually.
// 

#define HPX_OPENCL_OVERLOAD_FUNCTION(classname, funcname, args,             \
                                                     args_without_types)    \
                                                                            \
/* Creates an empty list and proxies to the normal function */              \
hpx::lcos::future<hpx::opencl::event>                                \
classname::funcname(args) const                                             \
{                                                                           \
    std::vector<hpx::opencl::event> events(0);                              \
    return funcname(args_without_types, events);                            \
}                                                                           \
                                                                            \
/* Creates a list with just one element and proxies */                      \
hpx::lcos::future<hpx::opencl::event>                                \
classname::funcname(args, hpx::opencl::event event) const                   \
{                                                                           \
    std::vector<hpx::opencl::event> events;                                 \
    events.push_back(event);                                                \
    return funcname(args_without_types, events);                            \
}                                                                           \
                                                                            \
/* This is the static async callback for a single future. */                \
/* Will get called by future.then().                      */                \
static                                                                      \
hpx::lcos::future<hpx::opencl::event>                                \
classname ## _ ## funcname ## _future_single_callback(classname cl, args,   \
                        hpx::lcos::shared_future<hpx::opencl::event> event) \
{                                                                           \
                                                                            \
    return cl.funcname(args_without_types, event.get());                    \
                                                                            \
}                                                                           \
                                                                            \
hpx::lcos::future<hpx::opencl::event>                                \
classname::funcname(args,                                                   \
                hpx::lcos::shared_future<hpx::opencl::event> event) const   \
{                                                                           \
    return event.then(                                                      \
            hpx::util::bind(                                                \
                    &(classname ## _ ## funcname ##                         \
                                  _future_single_callback),                 \
                    *this,                                                  \
                    args_without_types,                                     \
                    util::placeholders::_1                                  \
            )                                                               \
    );                                                                      \
}                                                                           \
                                                                            \
static                                                                      \
hpx::lcos::future<hpx::opencl::event>                                \
classname ## _ ## funcname ## _future_multi_callback(classname cl, args,    \
           hpx::lcos::future<std::vector<                            \
                        hpx::lcos::shared_future<hpx::opencl::event>        \
                                                            >> futures)     \
{                                                                           \
                                                                            \
    /* Get list of futures */                                               \
    std::vector<hpx::lcos::shared_future<hpx::opencl::event>>               \
    futures_list = futures.get();                                           \
                                                                            \
    /* Create list of events */                                             \
    std::vector<hpx::opencl::event> events;                                 \
    events.reserve(futures_list.size());                                    \
                                                                            \
    /* Put events into list */                                              \
    BOOST_FOREACH(hpx::lcos::shared_future<hpx::opencl::event> & future,    \
                    futures_list)                                           \
    {                                                                       \
        events.push_back(future.get());                                     \
    }                                                                       \
                                                                            \
    /* Call actual function */                                              \
    return cl.funcname(args_without_types, events);                         \
                                                                            \
}                                                                           \
                                                                            \
hpx::lcos::future<hpx::opencl::event>                                \
classname::funcname(args,                                                   \
        std::vector<hpx::lcos::shared_future<hpx::opencl::event>> events)   \
                                                                    const   \
{                                                                           \
                                                                            \
    return hpx::when_all(events).then(                                      \
        hpx::util::bind(                                                    \
                *(classname ## _ ## funcname ##                             \
                              _future_multi_callback),                      \
                *this,                                                      \
                args_without_types,                                         \
                util::placeholders::_1                                      \
        )                                                                   \
     );                                                                     \
                                                                            \
}



// Necessary for variable argument length
#define COMMA ,






#endif

