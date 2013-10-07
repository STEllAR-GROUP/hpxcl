// Copyright (c)	2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_TOOLS_HPP__
#define HPX_OPENCL_TOOLS_HPP__

#include <CL/cl.hpp>

namespace hpx { namespace opencl {

// To be called on OpenCL errorcodes, throws an exception on OpenCL Error
void clEnsure(cl_int errCode, const char* functionname);

// Translates CL errorcode to descriptive string
const char* clErrToStr(cl_int errCode);

}}

#endif//HPX_OPENCL_TOOLS_HPP__

