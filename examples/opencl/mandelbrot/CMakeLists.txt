# Copyright (c) 2013 Martin Stumpf
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(PNG_FOUND)

set(sources_common
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
    




source_group("Source Files" FILES ${sources_common} ${sources_main} ${sources_maps})





# add example executable
add_hpx_executable(mandelbrot
                   SOURCES ${sources_main} ${sources_common}
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
                            mandelbrot_exe)


# add example executable
add_hpx_executable(mandelbrot_maps
                   SOURCES ${sources_maps} ${sources_common}
                   DEPENDENCIES opencl_component
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
							
							
target_link_libraries(mandelbrot_exe ${PNG_LIBRARIES})
target_link_libraries(mandelbrot_maps_exe ${PNG_LIBRARIES})


endif()