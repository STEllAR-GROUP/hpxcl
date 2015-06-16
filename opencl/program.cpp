// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Header File
#include "program.hpp"

// Internal Dependencies
#include "server/program.hpp"

#include "kernel.hpp"

using hpx::opencl::program;

hpx::lcos::future<void>
program::build() const
{
    return build("");
}

hpx::lcos::future<void>
program::build(std::string build_options) const
{
    HPX_ASSERT(this->get_gid());

    typedef hpx::opencl::server::program::build_action func;

    return async<func>(this->get_gid(), build_options);
}

hpx::lcos::future<hpx::serialization::serialize_buffer<char> >
program::get_binary() const
{
    HPX_ASSERT(this->get_gid());

    typedef hpx::opencl::server::program::get_binary_action func;

    return async<func>(this->get_gid());
}

hpx::opencl::kernel
program::create_kernel(std::string kernel_name) const
{

    HPX_ASSERT(this->get_gid());

    typedef hpx::opencl::server::program::create_kernel_action func;
    
    hpx::future<hpx::id_type> kernel_server =
                                 hpx::async<func>(this->get_gid(), kernel_name);

    return kernel(std::move(kernel_server), device_gid);

}


