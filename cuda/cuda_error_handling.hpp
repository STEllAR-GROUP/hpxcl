// Copyright (c)        2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef CUDA_CUDA_CUDAERRORHANDLING_HPP_
#define CUDA_CUDA_CUDAERRORHANDLING_HPP_

#include <hpx/hpx.hpp>

#include "cuda/export_definitions.hpp"

#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

namespace hpx { namespace cuda
{

/** \brief Handles the error checking for CUDA functions calls
 *	and kernel executions
 *
 * This method checks if there is a cudaGetLastError and passes
 * the error found inside CUDA to and raises a hpx exception with the
 * cudaError code and the cudaErrorMessage.
 *
 * Inside of the hpx CUDA component this method is called after each CUDA
 * related function. In the raised error the class and the function where
 * the CUDA error was raised is givenas an additional information.
 *
 * \@param function_name name of the related cuda function or kernel execution
 *
 */
HPX_CUDA_EXPORT void checkCudaError(char const* function_name = "");

}}

#endif /* CUDA_CUDA_CUDAERRORHANDLING_HPP_ */
