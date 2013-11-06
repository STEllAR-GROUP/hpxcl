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

using namespace hpx::opencl;

void
program::build()
{
    return build_async("").get();
}

void
program::build(std::string build_options)
{
    return build_async(build_options).get();
}

hpx::lcos::future<void>
program::build_async()
{
    return build_async("");
}

hpx::lcos::future<void>
program::build_async(std::string build_options)
{

   BOOST_ASSERT(this->get_gid());

   typedef hpx::opencl::server::program::build_action func;

   return hpx::async<func>(this->get_gid(), build_options);

}
