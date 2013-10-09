// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_NAME_DEFINITIONS_HPP__
#define HPX_OPENCL_NAME_DEFINITIONS_HPP__


////////////////////////////////////////////////////////////////
namespace hpx { namespace opencl{
    
    ////////////////////////////////////////////////////////
    /// Defines costum opencl variables, as the original
    /// are pointers and not serializable
    ///
    typedef intptr_t clx_device_id;
    typedef intptr_t clx_platform_id;

}}

#endif
