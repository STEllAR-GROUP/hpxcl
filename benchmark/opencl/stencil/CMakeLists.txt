# Copyright (c) 2017 Madhavan Seshadri
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPXCL_WITH_NAIVE_OPENCL_BENCHMARK)
	############################################################################
	# Configure pure OPENCL benchmarks
	############################################################################
	
	link_directories(${OpenCL_LIBRARY})
	add_executable (stencil_opencl stencil.c)
	target_include_directories (stencil_opencl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()