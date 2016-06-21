// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_UTIL_GENERIC_BUFFER_HPP_
#define HPX_OPENCL_UTIL_GENERIC_BUFFER_HPP_

// Default includes
#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

// Export definitions
#include "../export_definitions.hpp"

namespace hpx {
namespace opencl {
namespace util {

        namespace detail{

            template<typename T>
            struct generic_buffer_impl{
                typedef hpx::future<hpx::serialization::serialize_buffer<char> >
                    data_type;
                public:
                    static hpx::future<T>
                    get(data_type && data){
                        return data.then(
                            [] (data_type && data) -> T
                            {
                                hpx::serialization::serialize_buffer<char>
                                raw_data = data.get();

                                // Compare lengths
                                HPX_ASSERT(sizeof(T) == raw_data.size());

                                return * reinterpret_cast<T*>(raw_data.data());
                            });
                    };
            };

            template<typename T>
            struct generic_buffer_impl<std::vector<T>>{
                typedef hpx::future<hpx::serialization::serialize_buffer<char> >
                    data_type;
                public:
                    static hpx::future<std::vector<T>>
                    get(data_type && data){
                        return data.then(
                            [] (data_type && data) -> std::vector<T>
                            {

                                hpx::serialization::serialize_buffer<char>
                                raw_data = data.get();

                                // Compute number of elements
                                std::size_t num_elements = raw_data.size() / sizeof(T);

                                // Initialize result vector
                                std::vector<T> result;
                                result.reserve(num_elements);

                                // Fill result vector
                                for(std::size_t i = 0; i + sizeof(T) <= raw_data.size();
                                    i+=sizeof(T))
                                {
                                    result.push_back(
                                        *reinterpret_cast<T*>(&raw_data.data()[i]) );
                                }

                                /* Compare lengths */
                                HPX_ASSERT(result.size() == num_elements);

                                return result;
                            });
                    };
            };

            template<>
            struct generic_buffer_impl<std::string>{
                typedef hpx::future<hpx::serialization::serialize_buffer<char> >
                    data_type;
                public:
                    HPX_OPENCL_EXPORT static hpx::future<std::string>
                        get(data_type && data);
            };
    };

    /////////////////////////////////////////
    /// @brief An accelerator device.
    ///
    class HPX_OPENCL_EXPORT generic_buffer
    {
        typedef hpx::future<hpx::serialization::serialize_buffer<char> >
            data_type;
        public:
            generic_buffer(data_type && data_) : data(std::move(data_)){}

            /**
             * @brief Converts the info to a generic datatype.
             *
             * @return The converted result
             */
            template <typename T>
            hpx::future<T> get()
            {
                return detail::generic_buffer_impl<T>::get(std::move(data));
            }

        private:
            data_type data;

    };

}}}


#endif// HPX_OPENCL_UTIL_GENERIC_BUFFER_HPP_


