// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(KERNEL_3_HPP)
#define KERNEL_3_HPP

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/include/async.hpp>

#include "../server/kernel.hpp"

namespace hpx
{
    namespace cuda
    {
        namespace stubs
        {
            struct kernel
                : hpx::components::stub_base<server::kernel>
            {
                static hpx::lcos::future<void> set_context_async(hpx::naming::id_type const& gid)
                {
                    typedef server::kernel::set_context_action action_type;
                    return hpx::async<action_type>(gid);
                }

            	static void set_context(hpx::naming::id_type const& gid)
                {
                    set_context_async(gid).get();
                }

                static hpx::lcos::future<void> set_stream_async(hpx::naming::id_type const& gid)
                {
                    typedef server::kernel::set_stream_action action_type;
                    return hpx::async<action_type>(gid);
                }

                static void set_stream(hpx::naming::id_type const& gid)
                {
                    set_stream_async(gid).get()
                }

                static hpx::lcos::future<void> set_diminsions_async(hpx::naming::id_type const& gid)
                {
                    typedef server::kernel::set_diminsions_action action_type;
                    return hpx::async<action_type>(gid);
                }

                static void set_diminsions(hpx::naming::id_type const& gid)
                {
                    set_diminsions(gid).get();
                }

                static hpx::lcos::future<void> set_args_async(hpx::naming::id_type const& gid)
                {
                    typedef server::kernel::set_args_action action_type;
                    return hpx::async<action_type>(gid);
                }

                static void set_args(hpx::naming::id_type const& gid)
                {
                    set_args_async(gid).get();
                }
            };
        }
    }
}
#endif
