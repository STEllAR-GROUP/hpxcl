// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(PROGRAM_3_HPP)
#define PROGRAM_3_HPP

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/include/async.hpp>

#include "../server/program.hpp"

namespace hpx
{
    namespace cuda
    {
        namespace stubs
        {
            struct program
                : hpx::components::stub_base<server::program>
            {
                static hpx::lcos::future<void> build(hpx::naming::id_type const &gid)
                {
                    typedef server::program::build_action action_type;
                    return hpx::async<action_type>(gid);
                }

                static void build_sync(hpx::naming::id_type const &gid)
                {
                    build(gid);
                }
            };
        }
    }
}
#endif //PROGRAM_3_HPP
