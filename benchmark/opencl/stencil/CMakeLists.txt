# Copyright (c) 2017 Madhavan Seshadri
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPXCL_WITH_NAIVE_OPENCL_BENCHMARK)
	############################################################################
	# Configure pure OPENCL benchmarks
	############################################################################
	
	include_directories(${OPENCL_INCLUDE_DIRS})

	link_directories(${OpenCL_LIBRARY})
	add_executable (stencil_opencl stencil.c)
	add_executable (partition_opencl partition.c)
	target_link_libraries(stencil_opencl m)
	target_link_libraries(partition_opencl m)
	target_include_directories (stencil_opencl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
	target_include_directories (partition_opencl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()