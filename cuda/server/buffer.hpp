// Copyright (c)		2013 Damond Howard
//						2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#pragma once
#ifndef HPX_CUDA_SERVER_BUFFER_HPP_
#define HPX_CUDA_SERVER_BUFFER_HPP_

#include <hpx/hpx.hpp>

#include <cuda.h>

#include "cuda/fwd_declarations.hpp"
#include "cuda/export_definitions.hpp"
#include "cuda/cuda_error_handling.hpp"
namespace hpx
{
    namespace cuda
    {
        namespace server
        {
            //////////////////////////////////////////////////////////
            ///This class represents a buffer of cuda kernel arguments

            class HPX_CUDA_EXPORT buffer
                : public hpx::components::locking_hook<
                    hpx::components::managed_component_base<buffer>
                    >
            {

                private:
                size_t arg_buffer_size;
                int parent_device_num;
                void* data_device;
                void* data_host;
                cudaStream_t stream;

                public:
                buffer();

                buffer(size_t size, int parent_device_num);

                size_t size();



                void set_size(size_t size);

                ~buffer();

                hpx::serialization::serialize_buffer<char>
                enqueue_read(size_t offset, size_t size);

                void enqueue_write(size_t offset, hpx::serialization::serialize_buffer<char> data);

                void* get_raw_pointer();

                HPX_DEFINE_COMPONENT_ACTION(buffer, size);
                HPX_DEFINE_COMPONENT_ACTION(buffer, set_size);
                HPX_DEFINE_COMPONENT_ACTION(buffer, enqueue_read);
                HPX_DEFINE_COMPONENT_ACTION(buffer, enqueue_write);
            };
        }
    }
}

 HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::buffer::size_action,
    buffer_size_action);
 HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::buffer::set_size_action,
    buffer_set_size_action);
 HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::buffer::enqueue_read_action,
    buffer_enqueue_read_action);
 HPX_REGISTER_ACTION_DECLARATION(
    hpx::cuda::server::buffer::enqueue_write_action,
    buffer_enqueue_write_action);

 #endif //BUFFER_2_HPP
