// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_FUTURE_EXECUTION_HPP__
#define HPX_OPENCL_FUTURE_EXECUTION_HPP__

#include <hpx/lcos/future.hpp>

#include "event.hpp"

#include <vector>

// This file is a general solution for all enqueue calls with
// future<event> input arguments.
//
// The general idea is to create an asynchronious function that immediately
// returns a future<event>, and then waits for the given future<event>s to
// complete. (Client-sided stacked asynchronism)

namespace hpx { namespace opencl {

    std::vector<event>
    wait_for_futures(const std::vector<hpx::lcos::future<event>> &future_list);


    #define future_call_def_0(PCLASS, FUNCNAME)                                \
                                                                               \
    class future_call{                                                         \
        private:                                                               \
        static event future_call_impl(PCLASS pclass,                           \
                          std::vector<hpx::lcos::future<event>> future_list)   \
        {                                                                      \
            /* Wait for futures to trigger */                                  \
            std::vector<event> event_list =                                    \
                    hpx::opencl::wait_for_futures(future_list);                \
                                                                               \
            /* Call desired function */                                        \
            return pclass.FUNCNAME(event_list).get();                          \
        }                                                                      \
        public:                                                                \
        static future<event> run(PCLASS pclass,                                \
                             std::vector<hpx::lcos::future<event>> future_list)\
        {                                                                      \
            /* Call future_call_impl assynchroniously */                       \
            return hpx::async(hpx::util::bind(future_call_impl, pclass,        \
                                              future_list));                   \
        }                                                                      \
    }


    #define future_call_def_1(PCLASS, ARG0, FUNCNAME)                          \
                                                                               \
    class future_call{                                                         \
        private:                                                               \
        static event future_call_impl(PCLASS pclass,                           \
                          ARG0 arg0,                                           \
                          std::vector<hpx::lcos::future<event>> future_list)   \
        {                                                                      \
            /* Wait for futures to trigger */                                  \
            std::vector<event> event_list =                                    \
                    hpx::opencl::wait_for_futures(future_list);                \
                                                                               \
            /* Call desired function */                                        \
            return pclass.FUNCNAME(arg0, event_list).get();                    \
        }                                                                      \
        public:                                                                \
        static future<event> run(PCLASS pclass,                                \
                                 ARG0 arg0,                                    \
                             std::vector<hpx::lcos::future<event>> future_list)\
        {                                                                      \
            /* Call future_call_impl assynchroniously */                       \
            return hpx::async(hpx::util::bind(future_call_impl, pclass,        \
                                              arg0,                            \
                                              future_list));                   \
        }                                                                      \
    }


    #define future_call_def_2(PCLASS, ARG0, ARG1, FUNCNAME)                    \
                                                                               \
    class future_call{                                                         \
        private:                                                               \
        static event future_call_impl(PCLASS pclass,                           \
                          ARG0 arg0,                                           \
                          ARG1 arg1,                                           \
                          std::vector<hpx::lcos::future<event>> future_list)   \
        {                                                                      \
            /* Wait for futures to trigger */                                  \
            std::vector<event> event_list =                                    \
                    hpx::opencl::wait_for_futures(future_list);                \
                                                                               \
            /* Call desired function */                                        \
            return pclass.FUNCNAME(arg0, arg1, event_list).get();              \
        }                                                                      \
        public:                                                                \
        static future<event> run(PCLASS pclass,                                \
                                 ARG0 arg0,                                    \
                                 ARG1 arg1,                                    \
                             std::vector<hpx::lcos::future<event>> future_list)\
        {                                                                      \
            /* Call future_call_impl assynchroniously */                       \
            return hpx::async(hpx::util::bind(future_call_impl, pclass,        \
                                              arg0,                            \
                                              arg1,                            \
                                              future_list));                   \
        }                                                                      \
    }


    #define future_call_def_3(PCLASS, ARG0, ARG1, ARG2, FUNCNAME)              \
                                                                               \
    class future_call{                                                         \
        private:                                                               \
        static event future_call_impl(PCLASS pclass,                           \
                          ARG0 arg0,                                           \
                          ARG1 arg1,                                           \
                          ARG2 arg2,                                           \
                          std::vector<hpx::lcos::future<event>> future_list)   \
        {                                                                      \
            /* Wait for futures to trigger */                                  \
            std::vector<event> event_list =                                    \
                    hpx::opencl::wait_for_futures(future_list);                \
                                                                               \
            /* Call desired function */                                        \
            return pclass.FUNCNAME(arg0, arg1, arg2, event_list).get();        \
        }                                                                      \
        public:                                                                \
        static future<event> run(PCLASS pclass,                                \
                                 ARG0 arg0,                                    \
                                 ARG1 arg1,                                    \
                                 ARG2 arg2,                                    \
                             std::vector<hpx::lcos::future<event>> future_list)\
        {                                                                      \
            /* Call future_call_impl assynchroniously */                       \
            return hpx::async(hpx::util::bind(future_call_impl, pclass,        \
                                              arg0,                            \
                                              arg1,                            \
                                              arg2,                            \
                                              future_list));                   \
        }                                                                      \
    }


    #define future_call_def_4(PCLASS, ARG0, ARG1, ARG2, ARG3, FUNCNAME)        \
                                                                               \
    class future_call{                                                         \
        private:                                                               \
        static event future_call_impl(PCLASS pclass,                           \
                          ARG0 arg0,                                           \
                          ARG1 arg1,                                           \
                          ARG2 arg2,                                           \
                          ARG3 arg3,                                           \
                          std::vector<hpx::lcos::future<event>> future_list)   \
        {                                                                      \
            /* Wait for futures to trigger */                                  \
            std::vector<event> event_list =                                    \
                    hpx::opencl::wait_for_futures(future_list);                \
                                                                               \
            /* Call desired function */                                        \
            return pclass.FUNCNAME(arg0, arg1, arg2, arg3, event_list).get();  \
        }                                                                      \
        public:                                                                \
        static future<event> run(PCLASS pclass,                                \
                                 ARG0 arg0,                                    \
                                 ARG1 arg1,                                    \
                                 ARG2 arg2,                                    \
                                 ARG3 arg3,                                    \
                             std::vector<hpx::lcos::future<event>> future_list)\
        {                                                                      \
            /* Call future_call_impl assynchroniously */                       \
            return hpx::async(hpx::util::bind(future_call_impl, pclass,        \
                                              arg0,                            \
                                              arg1,                            \
                                              arg2,                            \
                                              arg3,                            \
                                              future_list));                   \
        }                                                                      \
    }

    #define future_call_def_5(PCLASS, ARG0, ARG1, ARG2, ARG3, ARG4, FUNCNAME)  \
                                                                               \
    class future_call{                                                         \
        private:                                                               \
        static event future_call_impl(PCLASS pclass,                           \
                          ARG0 arg0,                                           \
                          ARG1 arg1,                                           \
                          ARG2 arg2,                                           \
                          ARG3 arg3,                                           \
                          ARG4 arg4,                                           \
                          std::vector<hpx::lcos::future<event>> future_list)   \
        {                                                                      \
            /* Wait for futures to trigger */                                  \
            std::vector<event> event_list =                                    \
                    hpx::opencl::wait_for_futures(future_list);                \
                                                                               \
            /* Call desired function */                                        \
            return pclass.FUNCNAME(arg0, arg1, arg2,                           \
                                   arg3, arg4, event_list).get();              \
        }                                                                      \
        public:                                                                \
        static future<event> run(PCLASS pclass,                                \
                                 ARG0 arg0,                                    \
                                 ARG1 arg1,                                    \
                                 ARG2 arg2,                                    \
                                 ARG3 arg3,                                    \
                                 ARG4 arg4,                                    \
                             std::vector<hpx::lcos::future<event>> future_list)\
        {                                                                      \
            /* Call future_call_impl assynchroniously */                       \
            return hpx::async(hpx::util::bind(future_call_impl, pclass,        \
                                              arg0,                            \
                                              arg1,                            \
                                              arg2,                            \
                                              arg3,                            \
                                              arg4,                            \
                                              future_list));                   \
        }                                                                      \
    }


}}


#endif

