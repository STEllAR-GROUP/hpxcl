// Copyright (c)		2013 Damond Howard
//						2015 patrick Diehl
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt
#include "tools.hpp"

#include <hpx/hpx.hpp>
#include <cuda.h>
#include <sstream>

namespace hpx
{
	namespace cuda
	{
		const char* cu_err_to_string(CUresult cu_error)
		{
			switch(cu_error)
			{
				case 0 : return "CUDA_SUCCESS";
				case 1 : return "CUDA_ERROR_INVALID VALUE";
				case 2 : return "CUDA_ERROR_OUT_OF_MEMORY";
				case 3 : return "CUDA_ERROR_NOT_INITIALIZED";
				case 4 : return "CUDA_ERROR_DEINITIALIZED";
				case 5 : return "CUDA_ERROR_PROFILER_DISABLED";
				case 6 : return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
				case 7 : return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
				case 8 : return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
				case 100 : return "CUDA_ERROR_NO_DEVICE";
				case 101 : return "CUDA_ERROR_INVALID_DEVICE";
				case 200 : return "CUDA_ERROR_INVALID_IMAGE";
				case 201 : return "CUDA_ERROR_INVALID_CONTEXT";
				case 202 : return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
				case 205 : return "CUDA_ERROR_MAP_FAILED";
				case 206 : return "CUDA_ERROR_UNMAP_FAILED";
				case 207 : return "CUDA_ERROR_ARRAY_IS_MAPPED";
				case 208 : return "CUDA_ERROR_ALREADY_MAPPED";
				case 209 : return "CUDA_ERROR_NO_BINARY_FOR_GPU";
				case 210 : return "CUDA_ERROR_ALREADY_AQUIRED";
				case 211 : return "CUDA_ERROR_NOT_MAPPED";
				case 212 : return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
				case 213 : return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
				case 214 : return "CUDA_ERROR_ECC_UNCORRECTABLE";
				case 215 : return "CUDA_ERROR_UNSUPPORTED_LIMIT";
				case 216 : return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
				case 217 : return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
				//case 218 : return "CUDA_ERROR_INVALID_PTX";
				case 300 : return "CUDA_ERROR_INVALID_SOURCE";
				case 301 : return "CUDA_ERROR_FILE_NOT_FOUND";
				case 302 : return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
				case 303 : return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
				case 304 : return "CUDA_ERROR_OPERATING_SYSTEM";
				case 400 : return "CUDA_ERROR_INVALID_HANDLE";
				case 500 : return "CUDA_ERROR_NOT_FOUND";
				case 600 : return "CUDA_ERROR_NOT_READY";
				case 700 : return "CUDA_ERROR_LAUNCH_FAILED";
				case 701 : return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCE";
				case 702 : return "CUDA_ERROR_LAUNCH_TIMEOUT";
				case 703 : return "CUDA_ERROR_LAUNCH_INCOMPLETE_TEXTURING";
				case 704 : return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
				case 705 : return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
				case 708 : return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
				case 709 : return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
				case 710 : return "CUDA_ERROR_ASSERT";
				case 711 : return "CUDA_ERROR_TOO_MANY_PEERS";
				case 712 : return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
				case 713 : return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
				//case 714 : return "CUDA_ERROR_HARDWARE_STACK_ERROR";
				//case 715 : return "CUDA_ERROR_ILLEGAL_INSTRUCTION";
				//case 716 : return "CUDA_ERROR_MISALIGNED_ADDRESS";
				//case 717 : return "CUDA_ERROR_INVALID_ADDRESS_SPACE";
				//case 718 : return "CUDA_ERROR_INVALID_PC";
				//case 719 : return "CUDA_ERROR_LAUNCH_FAILED";
				case 800 : return "CUDA_ERROR_NOT_PERMITTED";
				case 801 : return "CUDA_ERROR_NOT_SUPPORTED";
				case 999 : return "CUDA_ERROR_UNKNOWN";
				default	 : return "CUDA_ERROR_NOT_FOUND";
			}
		}
	}
}
