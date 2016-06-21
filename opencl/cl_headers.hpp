// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_CL_HEADERS_HPP_
#define HPX_OPENCL_CL_HEADERS_HPP_


#if defined(__APPLE__) || defined(__MACOSX)

//#include <OpenGL/OpenGL.h>
#include <OpenCL/opencl.h>

#else

//#include <GL/gl.h>
#include <CL/opencl.h>

#endif // !__APPLE__


#endif// HPX_OPENCL_CL_HEADERS_HPP_


