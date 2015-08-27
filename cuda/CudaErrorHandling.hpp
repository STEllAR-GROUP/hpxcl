/*
 * CudaErrorhandling.hpp
 *
 *  Created on: Aug 25, 2015
 *      Author: diehl
 */

#ifndef CUDA_CUDA_CUDAERRORHANDLING_HPP_
#define CUDA_CUDA_CUDAERRORHANDLING_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include <sstream>
#include <cuda.h>

#define checkCudaError  \
cudaError_t err = cudaGetLastError(); \
if (cudaSuccess != err) { \
	std::stringstream errorMessage; \
	errorMessage << "CudaError: " << cudaGetErrorString(err) << " in "\
	<< __FILE__ << " in line " << __LINE__ << std::endl;\
	HPX_THROW_EXCEPTION(hpx::no_success, "" ,errorMessage.str().c_str());\
	}

#endif /* CUDA_CUDA_CUDAERRORHANDLING_HPP_ */
