// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(PROGRAM_3_HPP)
#define PROGRAM_3_HPP

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/include/async.hpp>

#include <string>

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
                static hpx::lcos::future<void> build(hpx::naming::id_type const &gid, std::string NVCC_FLAGS)
                {
                    typedef server::program::build_action action_type;
                    return hpx::async<action_type>(gid, NVCC_FLAGS);
                }

                static void build_sync(hpx::naming::id_type const &gid, std::string NVCC_FLAGS)
                {
                    build(gid, NVCC_FLAGS).get();
                }

                static hpx::lcos::future<void> create_kernel(hpx::naming::id_type const &gid, std::string module_name, std::string kernel_name)
                {
                    typedef server::program::create_kernel_action action_type;
                    return hpx::async<action_type>(gid, module_name, kernel_name);
                }

                static void create_kernel_sync(hpx::naming::id_type const &gid, std::string module_name, std::string kernel_name)
                {
                    create_kernel(gid, module_name, kernel_name).get();
                }

                static void set_source_sync(hpx::naming::id_type const &gid, std::string source)
                {
                    typedef server::program::set_source_action action_type;
                    hpx::async<action_type>(gid, source).get();
                }
            };
        }
    }
}
#endif //PROGRAM_3_HPP
