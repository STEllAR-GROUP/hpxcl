# Copyright (c) 2017 Madhavan Seshadri
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPXCL_WITH_NAIVE_OPENCL_BENCHMARK)
	############################################################################
	# Configure pure OPENCL benchmarks
	############################################################################
	
	add_executable (smvp_opencl smvpCL.c)
	target_link_libraries(smvp_opencl m ${OPENCL_LIBRARIES})
	target_include_directories (smvp_opencl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

set(sources
    smvpHPX.cpp
)

source_group("Source Files" FILES ${sources})

# add example executable
add_hpx_executable(smvpHPX
                   SOURCES ${sources}
                   DEPENDENCIES opencl_component
                   COMPONENT_DEPENDENCIES iostreams
                   FOLDER "Benchmark/opencl/smvp")

# add a custom target for this example
add_hpx_pseudo_target(examples.opencl.smvpHPX)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.opencl
                            examples.opencl.smvpHPX)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.opencl.smvpHPX
                            smvpHPX_exe)
