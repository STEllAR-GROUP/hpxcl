/*
 * CudaErrorhandling.hpp
 *
 *  Created on: Aug 25, 2015
 *      Author: diehl
 */

#ifndef CUDA_CUDA_CUDAERRORHANDLING_HPP_
#define CUDA_CUDA_CUDAERRORHANDLING_HPP_

#include <hpx/hpx.hpp>

#include "cuda/export_definitions.hpp"

#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

namespace hpx { namespace cuda
{

HPX_CUDA_EXPORT void checkCudaError(char const* function_name = "");

}}

#endif /* CUDA_CUDA_CUDAERRORHANDLING_HPP_ */
