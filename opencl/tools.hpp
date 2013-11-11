// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_TOOLS_HPP__
#define HPX_OPENCL_TOOLS_HPP__

#include <CL/cl.hpp>
#include <sstream>
#include <hpx/include/iostreams.hpp>

namespace hpx { namespace opencl {

    // Used to disable the empty constructor of classes
    #define CL_FORBID_EMPTY_CONSTRUCTOR(classname)                           \
        classname::classname()                                               \
        {                                                                    \
            HPX_THROW_EXCEPTION(hpx::no_success, #classname "()",              \
                    "Empty constructor is not defined!");                    \
        }

    // To be called on OpenCL errorcodes, throws an exception on OpenCL Error
    #define clEnsure(errCode, functionname){                    \
        if(errCode != CL_SUCCESS)                               \
        {                                                       \
            std::stringstream errorMessage;                     \
            errorMessage << "CL_ERROR("                         \
                         << (errCode)                           \
                         << "): "                               \
                         << clErrToStr(errCode);                \
            HPX_THROW_EXCEPTION(hpx::no_success,                \
                                (functionname),                 \
                                errorMessage.str().c_str());    \
        }                                                       \
    }
    
     // To be called on OpenCL errorcodes in destructors, does not throw
    #define clEnsure_nothrow(errCode, functionname){            \
        if(errCode != CL_SUCCESS)                               \
        {                                                       \
            hpx::cerr << (functionname)                         \
                      << ": CL_ERROR("                          \
                      << (errCode)                              \
                      << "): "                                  \
                      << clErrToStr(errCode)                    \
                      << hpx::endl;                             \
        }                                                       \
    }
    
    
    // Translates CL errorcode to descriptive string
    const char* clErrToStr(cl_int errCode);

}}

#endif//HPX_OPENCL_TOOLS_HPP__

