// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_ENQUEUE_OVERLOADS_HPP__
#define HPX_OPENCL_ENQUEUE_OVERLOADS_HPP__



/*
 * The purpose of this file is to define the overloads of all the functions
 * that follow the common theme:
 *
 *
 *          hpx::lcos::unique_future<event>
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
 *          hpx::lcos::unique_future<event>
 *          class::function([args...]);
 *
 *          hpx::lcos::unique_future<event>
 *          class::function([args...], event);
 *
 *          hpx::lcos::unique_future<event>
 *          class::function([args...], hpx::lcos::shared_future<event>);
 *
 *          hpx::lcos::unique_future<event>
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

#define OVERLOAD_FUNCTION(classname, funcname, args, args_without_types)    \
                                                                            \
/* Creates an empty list and proxies to the normal function */              \
hpx::lcos::unique_future<hpx::opencl::event>                                \
classname::funcname(args) const                                             \
{                                                                           \
    std::vector<hpx::opencl::event> events(0);                              \
    return funcname(args_without_types, events);                            \
}                                                                           \
                                                                            \
/* Creates a list with just one element and proxies */                      \
hpx::lcos::unique_future<hpx::opencl::event>                                \
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
hpx::lcos::unique_future<hpx::opencl::event>                                \
funcname ## _future_single_callback(buffer buf, args,                       \
                        hpx::lcos::shared_future<hpx::opencl::event> event) \
{                                                                           \
                                                                            \
    return buf.funcname(args_without_types, event.get());                   \
                                                                            \
}                                                                           \
                                                                            \
hpx::lcos::unique_future<hpx::opencl::event>                                \
classname::funcname(args,                                                   \
                hpx::lcos::shared_future<hpx::opencl::event> event) const   \
{                                                                           \
    return event.then(                                                      \
            hpx::util::bind(                                                \
                    &(funcname ## _future_single_callback),                 \
                    *this,                                                  \
                    args_without_types,                                     \
                    util::placeholders::_1)                                 \
            );                                                              \
}                                                                           \
                                                                            \
hpx::lcos::unique_future<hpx::opencl::event>                                \
classname::funcname(args,                                                   \
        std::vector<hpx::lcos::shared_future<hpx::opencl::event>> events)   \
                                                                    const   \
{                                                                           \
/*    BOOST_ASSERT(this->get_gid());                                        \
                                                                            \
    // define the async call                                                \
    future_call_def_2(buffer, size_t, size_t, enqueue_read);                \
                                                                            \
    // run the async call                                                   \
    return future_call::run(*this, offset, size, events);                   \
                                                                            \
*/                                                                          \
    return unique_future<hpx::opencl::event>();                             \
}



// Necessary for variable argument length
#define COMMA ,






#endif

