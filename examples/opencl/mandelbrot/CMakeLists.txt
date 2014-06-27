# Copyright (c) 2013 Martin Stumpf
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


set(sources_lib
    pngwriter.cpp
    workload.cpp
    mandelbrotworker.cpp
    mandelbrotworker_buffermanager.cpp
    image_generator.cpp
)

set(sources_main
    main.cpp
)

set(sources_maps
    maps/main_maps.cpp
    maps/webserver.cpp
    maps/requesthandler.cpp
    maps/maps_image_generator.cpp
#    main_maps.cpp
#    mongoose.cpp
#    maps_webserver.cpp
)
    




source_group("Source Files" FILES ${sources_lib} ${sources_main} ${sources_maps})

# add mandelbrot component
add_hpx_component(mandelbrot_lib
                   SOURCES ${sources_lib}
                   DEPENDENCIES opencl_component
                   COMPONENT_DEPENDENCIES iostreams
                   FOLDER "Examples/OpenCL/mandelbrot")

# add a custom target for this example
add_hpx_pseudo_target(examples.opencl.mandelbrot_lib)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.opencl
                            examples.opencl.mandelbrot_lib)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.opencl.mandelbrot_lib
                            mandelbrot_lib_component)

set_target_properties(mandelbrot_lib_component
    PROPERTIES
    LINK_FLAGS "-lpng")



# add example executable
add_hpx_executable(mandelbrot
                   SOURCES ${sources_main}
                   DEPENDENCIES opencl_component mandelbrot_lib_component
                   COMPONENT_DEPENDENCIES iostreams
                   FOLDER "Examples/OpenCL/mandelbrot")

# add a custom target for this example
add_hpx_pseudo_target(examples.opencl.mandelbrot)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.opencl
                            examples.opencl.mandelbrot)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.opencl.mandelbrot
                            mandelbrot_exe)


# add example executable
add_hpx_executable(mandelbrot_maps
                   SOURCES ${sources_maps}
                   DEPENDENCIES opencl_component mandelbrot_lib_component
                   COMPONENT_DEPENDENCIES iostreams
                   FOLDER "Examples/OpenCL/mandelbrot")

# add a custom target for this example
add_hpx_pseudo_target(examples.opencl.mandelbrot_maps)

# make pseudo-targets depend on master pseudo-target
add_hpx_pseudo_dependencies(examples.opencl
                            examples.opencl.mandelbrot_maps)

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.opencl.mandelbrot_maps
                            mandelbrot_maps_exe)

