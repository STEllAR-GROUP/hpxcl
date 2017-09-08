// Copyright (c)    2013 Damond Howard
//                  2015 Patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#include "cuda/cuda_error_handling.hpp"

namespace hpx { namespace cuda
{

void checkCudaError(char const* function) {

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {

        std::stringstream errorMessage;
        errorMessage << "CudaError: " << cudaGetErrorString(err) << " at " << function << std::endl;

        HPX_THROW_EXCEPTION(hpx::no_success, function,
                errorMessage.str().c_str());

    }

}

}}


