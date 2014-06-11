# Copyright (c) 2007-2013 Hartmut Kaiser
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
# Copyright (c) 2013      Martin Stumpf
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(tests
    initialization
    buffer_read_write
    events_and_futures
    kernel
    future_enqueues
    program_from_binary
   )


#set(async_continue_PARAMETERS LOCALITIES 2)
#set(promise_PARAMETERS THREADS_PER_LOCALITY 4)
set(events_and_futures_PARAMETERS THREADS_PER_LOCALITY 4)


foreach(test ${tests})
  set(sources
      ${test}.cpp)

  set(${test}_FLAGS ${${test}_FLAGS} DEPENDENCIES opencl_component)

  source_group("Source Files" FILES ${sources})

  # add example executable
  add_hpx_executable(${test}_test
                     SOURCES ${sources}
                     ${${test}_FLAGS}
                     FOLDER "Tests/Unit/OpenCL")

  add_hpx_unit_test("opencl" ${test} ${${test}_PARAMETERS})

  # add a custom target for this example
  add_hpx_pseudo_target(tests.unit.opencl.${test})

  # make pseudo-targets depend on master pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.opencl
                              tests.unit.opencl.${test})

  # add dependencies to pseudo-target
  add_hpx_pseudo_dependencies(tests.unit.opencl.${test}
                              ${test}_test_exe)
endforeach()

