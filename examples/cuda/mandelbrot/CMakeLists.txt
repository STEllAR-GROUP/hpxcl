# Copyright (c)       2017 Madhavan Seshadri
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(sources
    mandelbrot.cpp
)


source_group("Source Files" FILES ${sources})

# add example executable
add_hpx_executable(mandlbrot
                   SOURCES ${sources}
                   COMPONENT_DEPENDENCIES iostreams
                   FOLDER "Examples/cuda/mandelbrot")

# add a custom target for this example
add_hpx_pseudo_target(examples.cuda.mandelbrot)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.cuda
                            examples.cuda.mandelbrot)