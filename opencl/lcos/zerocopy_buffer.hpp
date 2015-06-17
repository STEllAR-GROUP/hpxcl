// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_LCOS_ZEROCOPY_BUFFER_HPP_
#define HPX_OPENCL_LCOS_ZEROCOPY_BUFFER_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

namespace hpx { namespace opencl { namespace lcos
{

    //----------------------------------------------------------------------------
    // A custom allocator which takes a pointer in its constructor and then returns
    // this pointer in response to any allocate request. It is here to try to fool
    // the hpx serialization into copying directly into a user provided buffer
    // without copying from a result into another buffer.
    //
    class zerocopy_buffer
    {
    public:
        zerocopy_buffer() BOOST_NOEXCEPT
          : pointer_(0), size_(0)
        {
            std::cout << "zerocopy_buffer(0-0)" << std::endl;
        }
    
        zerocopy_buffer(std::uintptr_t p, std::size_t size,
                        hpx::serialization::serialize_buffer<char> buffer)
          : pointer_(p), size_(size), buffer_(buffer)
        {

            std::cout << "zerocopy_buffer(" << size_ << ")" << std::endl;
            HPX_ASSERT(buffer.size() == size_);
            //HPX_ASSERT(data_() == static_cast<char*>(buffer.data()));

        }
        
        std::size_t size(){
            return size_;
        }
    
    private:
        // serialization support
        friend class hpx::serialization::access;
    
        template <typename Archive>
        void load(Archive& ar, unsigned int const version)
        {
            // read size and adress
            ar >> size_ >> pointer_;
            // write data to adress
            char* dest_addr = reinterpret_cast<char*>(pointer_);
            ar >> hpx::serialization::make_array(dest_addr, size_);
            std::cout << "zerocopy_buffer_load(" << size_ << ")" << std::endl;
        }
    
        template <typename Archive>
        void save(Archive& ar, unsigned int const version) const
        {
            std::cout << "zerocopy_buffer_save(" << size_ << ")" << std::endl;
            // send size, adress and data
            ar << size_ << pointer_
               << hpx::serialization::make_array(buffer_.data(), size_);
        }

        HPX_SERIALIZATION_SPLIT_MEMBER()
    
    private:
        std::uintptr_t pointer_;
        std::size_t size_;
        hpx::serialization::serialize_buffer<char> buffer_;
    };

}}}
#endif
