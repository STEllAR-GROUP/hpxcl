// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(BUFFER_1_HPP)
#define BUFFER_1_HPP

#include <hpx/include/components.hpp>
#include <hpx/runtime/serialization/serialize_buffer.hpp>
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

                hpx::lcos::future<void> set_size(size_t size)
                {
                    BOOST_ASSERT(this->get_gid());
                    return this->base_type::set_size(this->get_gid(), size);
                }

                void set_size_sync(size_t size)
                {
                    BOOST_ASSERT(this->get_gid());
                    this->base_type::set_size_sync(this->get_gid(), size);
                }

                void enqueue_read(size_t offset, size_t size) const
                {
                    BOOST_ASSERT(this->get_gid());

                    this->base_type::enqueue_read(this->get_gid(), offset, size);
                }
                
                void enqueue_write(size_t offset, size_t size, const void* data) const
                {
                    BOOST_ASSERT(this->get_gid());

                     hpx::serialization::serialize_buffer<char>
                        serializable_data((char*)const_cast<void*>(data), size,
                             hpx::serialization::serialize_buffer<char>::init_mode::reference);

                    this->base_type::enqueue_write(this->get_gid(), offset, serializable_data);
                }
        };
    }
}
#endif //BUFFER_1_HPP
