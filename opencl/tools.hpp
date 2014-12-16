// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_TOOLS_HPP_
#define HPX_OPENCL_TOOLS_HPP_

#include <CL/cl.h>
#include <sstream>

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>
#include <hpx/include/iostreams.hpp>

#ifndef CL_VERSION_1_1
#error "OpenCL 1.1 required!"
#endif

namespace hpx { namespace opencl {

    // Used to disable the empty constructor of classes
    #define CL_FORBID_EMPTY_CONSTRUCTOR(classname)                           \
        classname::classname()                                               \
        {                                                                    \
            HPX_THROW_EXCEPTION(hpx::no_success, #classname "()",            \
                    "Empty constructor is not defined!");                    \
        }

    // To be called on OpenCL errorcodes, throws an exception on OpenCL Error
    #define cl_ensure(errCode, functionname){                   \
        if(errCode != CL_SUCCESS)                               \
        {                                                       \
            std::stringstream errorMessage;                     \
            errorMessage << "CL_ERROR("                         \
                         << (errCode)                           \
                         << "): "                               \
                         << hpx::opencl::cl_err_to_str(errCode);\
            HPX_THROW_EXCEPTION(hpx::no_success,                \
                                (functionname),                 \
                                errorMessage.str().c_str());    \
        }                                                       \
    }
    
     // To be called on OpenCL errorcodes in destructors, does not throw
    #define cl_ensure_nothrow(errCode, functionname){           \
        if(errCode != CL_SUCCESS)                               \
        {                                                       \
            hpx::cerr << (functionname)                         \
                      << ": CL_ERROR("                          \
                      << (errCode)                              \
                      << "): "                                  \
                      << hpx::opencl::cl_err_to_str(errCode)    \
                      << hpx::endl;                             \
        }                                                       \
    }
    
    
    // Translates CL errorcode to descriptive string
    const char* cl_err_to_str(cl_int errCode);

}}

#endif//HPX_OPENCL_TOOLS_HPP_


