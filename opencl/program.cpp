// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Header File
#include "program.hpp"

// Internal Dependencies
#include "server/program.hpp"

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


