// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef HPX_OPENCL_EXPORT_DEFINITIONS_HPP_
#define HPX_OPENCL_EXPORT_DEFINITIONS_HPP_

#include <hpx/config.hpp>
#include <hpx/config/export_definitions.hpp>

#if defined(HPX_OPENCL_MODULE_EXPORTS)
#define HPX_OPENCL_EXPORT HPX_SYMBOL_EXPORT
#else
#define HPX_OPENCL_EXPORT HPX_SYMBOL_IMPORT
#endif

#endif  // HPX_OPENCL_EXPORT_DEFINITIONS_HPP_
