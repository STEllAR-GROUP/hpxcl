// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include "server/program.hpp"

#include "program.hpp"
#include "kernel.hpp"

using namespace hpx::opencl;

void
program::build() const
{
    return build_async("").get();
}

void
program::build(std::string build_options) const
{
    return build_async(build_options).get();
}

hpx::lcos::unique_future<void>
program::build_async() const
{
    return build_async("");
}

hpx::lcos::unique_future<void>
program::build_async(std::string build_options) const
{

    BOOST_ASSERT(this->get_gid());

    typedef hpx::opencl::server::program::build_action func;

    return hpx::async<func>(this->get_gid(), build_options);

}

hpx::opencl::kernel
program::create_kernel(std::string kernel_name) const
{

    BOOST_ASSERT(this->get_gid());

    // Create new kernel object server
    hpx::lcos::unique_future<hpx::naming::id_type>
    kernel_server = hpx::components::new_<hpx::opencl::server::kernel>
                    (get_colocation_id_sync(get_gid()), get_gid(), kernel_name);

    return hpx::opencl::kernel(std::move(kernel_server));

}

