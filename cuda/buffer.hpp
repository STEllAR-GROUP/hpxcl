// Copyright (c)    2013 Damond Howard
//                  2015 patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#ifndef HPX_CUDA_BUFFER_HPP_
#define HPX_CUDA_BUFFER_HPP_

#include <hpx/hpx.hpp>

#include "cuda/server/buffer.hpp"

namespace hpx {
namespace cuda {

class buffer: public hpx::components::client_base<buffer, cuda::server::buffer> {
    typedef hpx::components::client_base<buffer, cuda::server::buffer> base_type;

public:
    buffer() {
    }

    buffer(hpx::future<hpx::naming::id_type> && gid) :
            base_type(std::move(gid)) {
    }

    hpx::lcos::future<size_t> size() {
        BOOST_ASSERT(this->get_gid());
        typedef server::buffer::size_action action_type;
        return hpx::async < action_type > (this->get_gid());

    }

    size_t size_sync() {
        return size().get();

    }

    hpx::lcos::future<void> set_size(size_t size) {
        HPX_ASSERT(this->get_gid());
        typedef server::buffer::set_size_action action_type;
        return hpx::async < action_type > (this->get_gid(), size);

    }

    void set_size_sync(size_t size) {
        set_size(size).get();

    }

    void enqueue_read(size_t offset, size_t size) const {
        HPX_ASSERT(this->get_gid());

        typedef server::buffer::enqueue_read_action action_type;
        hpx::async < action_type > (this->get_gid(), offset, size);

    }

    void enqueue_write(size_t offset, size_t size, const void* data) const {
        HPX_ASSERT(this->get_gid());

        hpx::serialization::serialize_buffer<char> serializable_data(
                (char*) const_cast<void*>(data), size,
                hpx::serialization::serialize_buffer<char>::init_mode::reference);

        typedef server::buffer::enqueue_write_action action_type;
        hpx::async < action_type > (this->get_gid(), offset, serializable_data);

    }
};
}
}
#endif //BUFFER_1_HPP
