// Copyright (c)		2013 Damond Howard
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#if !defined(BUFFER_3_HPP)
#define BUFFER_3_HPP

#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/serialization.hpp>

#include "../server/buffer.hpp"

namespace hpx
{
    namespace cuda
    {
        namespace stubs
        {
            struct buffer
                : hpx::components::stub_base<server::kernel>
            {
            	static hpx::lcos::future<size_t> size(hpx::naming::id_type const& gid)
                {
                    typedef server::buffer::size_action action_type;
                    return hpx::async<action_type>(gid);
                }

                static size_t size_sync(hpx::naming::id_type const& gid)
                {
                    return size(gid).get();
                }

                static hpx::lcos::future<void> set_size(hpx::naming::id_type const& gid, size_t size)
                {
                    typedef server::buffer::set_size_action action_type;
                    return hpx::async<action_type>(gid, size);
                }

                static void set_size_sync(hpx::naming::id_type const& gid, size_t size)
                {
                    set_size(gid, size).get();
                }

                static hpx::lcos::future<void> enqueue_read(hpx::naming::id_type const& gid, size_t offset, size_t size)
                {
                    typedef server::buffer::enqueue_read_action action_type;
                    return hpx::async<action_type>(gid, offset, size);
                }

                static hpx::lcos::future<void> enqueue_write(hpx::naming::id_type const& gid, size_t offset, hpx::serialization::serialize_buffer<char> data)
                {
                    typedef server::buffer::enqueue_write_action action_type;
                    return hpx::async<action_type>(gid, offset, data);
                }
            };
        }
    }
}
#endif //BUFFER_3_HPP
