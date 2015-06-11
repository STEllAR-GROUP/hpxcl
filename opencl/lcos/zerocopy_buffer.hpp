// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_LCOS_ZEROCOPY_BUFFER_HPP_
#define HPX_OPENCL_LCOS_ZEROCOPY_BUFFER_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

namespace hpx { namespace opencl { namespace util
{

    //----------------------------------------------------------------------------
    // A custom allocator which takes a pointer in its constructor and then returns
    // this pointer in response to any allocate request. It is here to try to fool
    // the hpx serialization into copying directly into a user provided buffer
    // without copying from a result into another buffer.
    //
    class zerocopy_buffer
    {
    private:
        template <typename T>
        static char*
        get_data(hpx::serialization::serialize_buffer<T> buffer){
            return static_cast<char*>(buffer.data());
        }

    public:
        zerocopy_buffer() BOOST_NOEXCEPT
          : pointer_(0), size_(0)
        {
            std::cout << "zerocopy_buffer(0-0)" << std::endl;
        }
    
        // explanation: keep buffer alive by binding it to get_data().
        // this also strips away the template parameter.
        template<typename T>
        zerocopy_buffer(std::uintptr_t p, std::size_t size,
                        hpx::serialization::serialize_buffer<T> buffer)
          : pointer_(reinterpret_cast<T*>(p)), size_(size),
            data_(hpx::bind(get_data<T>, buffer));
        {

            std::cout << "zerocopy_buffer(" << size_ << ")" << std::endl;
            HPX_ASSERT(buffer_.size() * sizeof(T) == size_);
            HPX_ASSERT(data() == static_cast<char*>(buffer.data()));

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
            ar << size_ << pointer_ << make_array(data_(), size_);
        }
    
        HPX_SERIALIZATION_SPLIT_MEMBER()
    
    private:
        std::uintptr_t pointer_;
        std::size_t size_;
        hpx::util::function<char*(void)> data_;
    };

}}}
#endif
