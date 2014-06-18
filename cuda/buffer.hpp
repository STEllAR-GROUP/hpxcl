// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(BUFFER_1_HPP)
#define BUFFER_1_HPP

#include <hpx/include/components.hpp>
#include "stubs/kernel.hpp"

namespace hpx
{
    namespace cuda
    {
        class buffer
            : public hpx::components::client_base<
                buffer, stubs::buffer >
        {
            typedef hpx::components::client_base<
                buffer, stubs::buffer>
                base_type;

            public:
                buffer()
                {}

                buffer(hpx::future<hpx::naming::id_type> const& gid)
                : base_type(gid)
                {}

                size_t size()
                {
                    BOOST_ASSERT(this->get_gid());
                    return this->base_type::size(this->get_gid());
                }

                void push_back(void *arg)
                {
                    BOOST_ASSERT(this->get_gid());
                    this->base_type::push_back(this->get_gid(),arg);
                }

                void load_args()
                {
                    BOOST_ASSERT(this->get_gid());
                    this->base_type::load_config(this->get_gid());
                }
        };
    }
}
#endif //BUFFER_1_HPP
