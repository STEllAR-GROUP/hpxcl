# Copyright (c) 2013 Martin Stumpf
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


set(sources
    main.cpp
)


source_group("Source Files" FILES ${sources})

# add example executable
add_hpx_executable(mandelbrot
                   SOURCES ${sources}
                   DEPENDENCIES opencl_component
                   COMPONENT_DEPENDENCIES iostreams
                   FOLDER "Examples/OpenCL/mandelbrot")

# add a custom target for this example
add_hpx_pseudo_target(examples.opencl.mandelbrot)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.opencl
                            examples.opencl.mandelbrot)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.opencl.mandelbrot
                            mandelbrot)

