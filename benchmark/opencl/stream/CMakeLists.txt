# Copyright (c) 2017 Madhavan Seshadri
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPXCL_WITH_NAIVE_OPENCL_BENCHMARK)
	############################################################################
	# Configure pure OPENCL benchmarks
	############################################################################
	
	link_directories(${OpenCL_LIBRARY})
	add_executable (stream_opencl stream.c)
	target_include_directories (stream_opencl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()