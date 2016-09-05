// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_LCOS_ZEROCOPY_BUFFER_HPP_
#define HPX_OPENCL_LCOS_ZEROCOPY_BUFFER_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include "../util/rect_props.hpp"

namespace hpx { namespace opencl { namespace lcos
{
    //----------------------------------------------------------------------------
    // A custom allocator which takes a pointer in its constructor and then returns
    // this pointer in response to any allocate request. It is here to try to fool
    // the hpx serialization into copying directly into a user provided buffer
    // without copying from a result into another buffer.
    //
    class zerocopy_buffer
      : public hpx::serialization::serialize_buffer<char>
    {
        typedef hpx::serialization::serialize_buffer<char> base_type;

    public:
        zerocopy_buffer() BOOST_NOEXCEPT
          : pointer_(0), size_x(0), size_y(0), size_z(0)
        {
        }

        zerocopy_buffer(std::uintptr_t p, std::size_t size,
                        hpx::serialization::serialize_buffer<char> buffer)
          : base_type(buffer), pointer_(p), size_x(size),
            size_y(1), size_z(1), stride_y(0), stride_z(0)
        {
            HPX_ASSERT(this->base_type::size() == size_x * size_y * size_z);
        }

        zerocopy_buffer( std::uintptr_t p,
                         const hpx::opencl::rect_props & rect,
                         std::size_t elem_size,
                         hpx::serialization::serialize_buffer<char> buffer)
          : base_type(buffer),
            pointer_(p),
            size_x(rect.size_x * elem_size),
            size_y(rect.size_y),
            size_z(rect.size_z),
            stride_y(rect.dst_stride_y * elem_size),
            stride_z(rect.dst_stride_z * elem_size)
        {
            // add origin position to pointer_. reduces network traffic
            // as dst_x, dst_y and dst_z don't need to be transmitted.
            pointer_ += rect.dst_x + stride_y*rect.dst_y + stride_z*rect.dst_z;

            HPX_ASSERT(this->base_type::size() == size_x * size_y * size_z);
        }

    private:
        // serialization support
        friend class hpx::serialization::access;

        template <typename Archive>
        void load(Archive& ar, unsigned int const version)
        {
            // deliberately don't serialize base class

            // read size and address
            ar >> size_x >> size_y >> size_z >> stride_y >> stride_z >> pointer_;

            // write data to address
            char* dest_addr = reinterpret_cast<char*>(pointer_);
            for(std::size_t z = 0; z < size_z; z++)
            {
                for(std::size_t y = 0; y < size_y; y++)
                {
                    ar >> hpx::serialization::make_array(
                            dest_addr + y * stride_y + z * stride_z,
                            size_x);
                }
            }
        }

        template <typename Archive>
        void save(Archive& ar, unsigned int const version) const
        {
            // deliberately don't serialize base class

            // send size, address and data
            ar << size_x << size_y << size_z << stride_y << stride_z << pointer_;
            ar << hpx::serialization::make_array(
                this->base_type::data(), this->base_type::size() );
        }

        HPX_SERIALIZATION_SPLIT_MEMBER()

    private:
        std::uintptr_t pointer_;
        std::size_t size_x;
        std::size_t size_y;
        std::size_t size_z;
        std::size_t stride_y;
        std::size_t stride_z;
        hpx::serialization::serialize_buffer<char> buffer_;
    };
}}}

#endif
