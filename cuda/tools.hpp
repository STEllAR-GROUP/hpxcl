// Copyright (c)    2013 Damond Howard
//                  2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#pragma once
#ifndef HPX_CUDA_TOOLS_HPP_
#define HPX_CUDA_TOOLS_HPP_

#include <cuda.h>
#include <sstream>

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

namespace hpx
{
    namespace cuda
    {
        // to be called on CUDA errorcodes, throws an exception on CUDA Error
        #define cuda_ensure(errCode, functionname)                            \
        {                                                                     \
            if(errCode != CUDA_SUCCESS){                                      \
                std::stringstream errorMessage;                               \
                errorMessage << "CUDA_ERROR("                                 \
                             << (errCode)                                     \
                             << "):"                                          \
                             << hpx::cuda::cuda_err_to_string(errCode);       \
                HPX_THROW_EXCEPTION(hpx::no_success,                          \
                                    (functionname),                           \
                                    errorMessage.str().c_str());              \
            }                                                                 \
        }                                                                     \

        const char* cu_err_to_string(CUresult cu_error);
    }
}
#endif //HPX_CUDA_TOOLS_HPP_
