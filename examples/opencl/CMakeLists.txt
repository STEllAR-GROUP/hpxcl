# Copyright (c) 2013 Martin Stumpf
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    hello_world
    _tmp_test
    events_and_futures
)

foreach(example_program ${example_programs})
  set(sources
      ${example_program}.cpp)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${example_program}
                     SOURCES ${sources}
                     DEPENDENCIES opencl_component
                     COMPONENT_DEPENDENCIES iostreams
                     FOLDER "Examples/OpenCL/${example_program}")

  # add a custom target for this example
  add_hpx_pseudo_target(examples.opencl.${example_program})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(examples.opencl
                              examples.opencl.${example_program})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(examples.opencl.${example_program}
                              ${example_program}_exe)
endforeach()

