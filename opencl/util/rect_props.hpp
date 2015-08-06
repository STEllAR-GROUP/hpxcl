// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_UTIL_RECT_PROPS_HPP_
#define HPX_OPENCL_UTIL_RECT_PROPS_HPP_

// Default includes
#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

// Export definitions
#include "../export_definitions.hpp"

namespace hpx {
namespace opencl { 

    //////////////////////////////////////
    /// @brief Metadata vector for _rect copy operations
    ///
    /// This structure is used for Rect data copy functions.
    /// 
    struct rect_props
    {
        public:
            std::size_t src_x;
            std::size_t src_y;
            std::size_t src_z;
            std::size_t dst_x;
            std::size_t dst_y;
            std::size_t dst_z;
            std::size_t size_x;
            std::size_t size_y;
            std::size_t size_z;
            std::size_t src_stride_y;
            std::size_t src_stride_z;
            std::size_t dst_stride_y;
            std::size_t dst_stride_z;
            rect_props( std::size_t src_x_,
                        std::size_t src_y_,
                        std::size_t src_z_,
                        std::size_t dst_x_,
                        std::size_t dst_y_,
                        std::size_t dst_z_,
                        std::size_t size_x_,
                        std::size_t size_y_,
                        std::size_t size_z_,
                        std::size_t src_stride_y_,
                        std::size_t src_stride_z_,
                        std::size_t dst_stride_y_,
                        std::size_t dst_stride_z_ )
                : src_x(src_x_),
                  src_y(src_y_),
                  src_z(src_z_),
                  dst_x(dst_x_),
                  dst_y(dst_y_),
                  dst_z(dst_z_),
                  size_x(size_x_),
                  size_y(size_y_),
                  size_z(size_z_),
                  src_stride_y(src_stride_y_),
                  src_stride_z(src_stride_z_),
                  dst_stride_y(dst_stride_y_),
                  dst_stride_z(dst_stride_z_){}
            rect_props( std::size_t src_x_,
                        std::size_t src_y_,
                        std::size_t dst_x_,
                        std::size_t dst_y_,
                        std::size_t size_x_,
                        std::size_t size_y_,
                        std::size_t src_stride_y_,
                        std::size_t dst_stride_y_)
                : src_x(src_x_),
                  src_y(src_y_),
                  src_z(0),
                  dst_x(dst_x_),
                  dst_y(dst_y_),
                  dst_z(0),
                  size_x(size_x_),
                  size_y(size_y_),
                  size_z(1),
                  src_stride_y(src_stride_y_),
                  src_stride_z(0),
                  dst_stride_y(dst_stride_y_),
                  dst_stride_z(0){}
            rect_props()
                : src_x(0),
                  src_y(0),
                  src_z(0),
                  dst_x(0),
                  dst_y(0),
                  dst_z(0),
                  size_x(1),
                  size_y(1),
                  size_z(1),
                  src_stride_y(0),
                  src_stride_z(0),
                  dst_stride_y(0),
                  dst_stride_z(0){}
        private:
            // serialization support
            friend class hpx::serialization::access;
    
            ///////////////////////////////////////////////////////////////////////
            template <typename Archive>
            void save(Archive& ar, const unsigned int version) const
            {
                ar << src_x;
                ar << src_y;
                ar << src_z;
                ar << dst_x;
                ar << dst_y;
                ar << dst_z;
                ar << size_x;
                ar << size_y;
                ar << size_z;
                ar << src_stride_y;
                ar << src_stride_z;
                ar << dst_stride_y;
                ar << dst_stride_z;
            }
    
            ///////////////////////////////////////////////////////////////////////
            template <typename Archive>
            void load(Archive& ar, const unsigned int version)
            {
                ar >> src_x;
                ar >> src_y;
                ar >> src_z;
                ar >> dst_x;
                ar >> dst_y;
                ar >> dst_z;
                ar >> size_x;
                ar >> size_y;
                ar >> size_z;
                ar >> src_stride_y;
                ar >> src_stride_z;
                ar >> dst_stride_y;
                ar >> dst_stride_z;
            }
    
            HPX_SERIALIZATION_SPLIT_MEMBER()
    };

}}


#endif// HPX_OPENCL_UTIL_RECT_PROPS_HPP_

            
