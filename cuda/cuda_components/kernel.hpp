// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(KERNEL_1_HPP)
#define KERNEL_1_HPP

#include <hpx/include/components.hpp>
#include "stubs/kernel.hpp"

namespace hpx
{
    namespace cuda
    {
        class kernel
            : public hpx::components::client_base<
                kernel, stubs::kernel >
        {
            typedef hpx::components::client_base<
                kernel, stubs::kernel>
                base_type;

            public:
                kernel()
                {}

                kernel(hpx::future<hpx::naming::id_type> const& gid)
                : base_type(gid)
                {}

                static hpx::lcos::future<void> set_context_async()
                { 
                    BOOST_ASSERT(this->get_gid());
                    return this->base_type::set_context_async(this->get_gid());
                }

                void set_context()
                {
                    BOOST_ASSERT(this->get_gid());
                    this->base_type::set_context(this->get_gid());
                }

                hpx::lcos::future<void> set_stream_async()
                {
                    BOOST_ASSERT(this->get_gid());
                    return this->base_type::set_stream_async(this->get_gid());
                }

                void set_stream()
                {
                    BOOST_ASSERT(this->get_gid());
                    this->base_type::set_stream(this->get_gid());
                }

                hpx::lcos::future<void> set_diminsions_async()
                {
                    BOOST_ASSERT(this->get_gid());
                    return this->base_type::set_diminsions_async(this->get_gid());
                }

                void set_diminsions()
                {
                    BOOST_ASSERT(this->get_gid());
                    this->base_type::set_diminsions(this->get_gid());
                }

                hpx::lcos::future<void> set_args_async()
                {
                    BOOST_ASSERT(this->get_gid());
                    return this->base_type::set::set_args_async(this->get_gid());
                }

                void set_args()
                {
                    BOOST_ASSERT(this->get_gid());
                    this->base_type::set_args(this->get_gid());
                }

        };
    }
}
#endif //KERNEL_1_HPP
