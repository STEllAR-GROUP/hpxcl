// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(BUFFER_1_HPP)
#define BUFFER_1_HPP

#include <hpx/include/components.hpp>
#include "stubs/buffer.hpp"

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

                buffer(hpx::future<hpx::naming::id_type> && gid)
                : base_type(std::move(gid))
                {}

                hpx::lcos::future<size_t> size()
                {
                    BOOST_ASSERT(this->get_gid());
                    return this->base_type::size(this->get_gid());
                }

                size_t size_sync()
                {
                    BOOST_ASSERT(this->get_gid());
                    return this->base_type::size_sync(this->get_gid());
                }

                /*void enqueue_read(size_t offset, size_t size) const
                {
                    BOOST_ASSERT(this->get_gid());
                    this->base_type::enqueue_read(this->get_gid());
                }

                void enqueue_write(size_t offset, size_t size, const void* data) const
                {
                    BOOST_ASSERT(this->get_gid());
                    this->base_type::enqueue_read(this->get_gid());
                }*/
        };
    }
}
#endif //BUFFER_1_HPP
