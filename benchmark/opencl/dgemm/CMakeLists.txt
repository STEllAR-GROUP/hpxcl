# Copyright (c) 2017 Madhavan Seshadri
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPXCL_WITH_NAIVE_OPENCL_BENCHMARK)
	############################################################################
	# Configure pure OPENCL benchmarks
	############################################################################
	
	add_executable (dgemm_opencl dgemmCL.c)
	target_link_libraries(dgemm_opencl m ${OPENCL_LIBRARIES})
	target_include_directories (dgemm_opencl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

set(sources
    dgemmHPX.cpp
)

source_group("Source Files" FILES ${sources})

# add example executable
add_hpx_executable(dgemmHPX
                   SOURCES ${sources}
                   DEPENDENCIES opencl_component
                   COMPONENT_DEPENDENCIES iostreams
                   FOLDER "Benchmark/opencl/dgemm")

# add a custom target for this example
add_hpx_pseudo_target(examples.opencl.dgemmHPX)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.opencl
                            examples.opencl.dgemmHPX)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.opencl.dgemmHPX
                            dgemmHPX_exe)
