// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(EVEMT_3_HPP)
#define EVENT_3_HPP

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/include/async.hpp>

#include "../sever/event.hpp"

namespace hpx
{
    namespace cuda
    {
        namespace stubs
        {
            struct event
                : hpx::components::stub_base<server::event>
            {
            	static hpx::lcos::future<void> await_async(hpx::naming::id_type const& gid)
                {
                    typedef server::event::await_action action_type;
                    hpx::async<action_type>(gid);
                }

                static void await(hpx::naming::id_type const& gid)
                {
                    await_async(gid).get();
                }

                static hpx::lcos::future<bool> finished_async(hpx::naming::id_type const& gid)
                {
                    typedef server::event::finished_action action_type;
                    hpx::async<action_type>(gid);
                }

                static bool finished(hpx::naming::id_type const& gid)
                {
                    finished_async(gid).get();
                }

                static hpx::lcos::future<void> trigger_async(hpx::naming::id_type const& gid)
                {
                    typedef server::event::trigger_action action_type;
                    hpx::async<action_type>(gid);
                }

                static void trigger(hpx::naming::id_type const& gid)
                {
                    trigger_async(gid).get();
                }
            };
        }
    }
}
#endif
