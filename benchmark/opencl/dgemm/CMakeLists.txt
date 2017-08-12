# Copyright (c) 2017 Madhavan Seshadri
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(sources
    dgemmHPX.cpp
)

source_group("Source Files" FILES ${sources})

# add example executable
add_hpx_executable(dgemmHPX
                   SOURCES ${sources}
                   DEPENDENCIES opencl_component
                   COMPONENT_DEPENDENCIES iostreams
                   FOLDER "benchmark/opencl/dgemm")

# add a custom target for this example
add_hpx_pseudo_target(examples.opencl.dgemmHPX)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.opencl
                            examples.opencl.dgemmHPX)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.opencl.dgemmHPX
                            dgemmHPX_exe)
