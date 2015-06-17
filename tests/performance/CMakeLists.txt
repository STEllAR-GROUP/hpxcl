# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
# Copyright (c) 2007-2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs "")

if(HPXCL_WITH_OPENCL)
  set(subdirs
    ${subdirs} opencl)
endif()

if(HPXCL_WITH_CUDA)
  set(subdirs
    ${subdirs} cuda)
endif()




if(NOT HPX_HAVE_SECURITY AND SODIUM_FOUND)
    set(subdirs ${subdirs} security)
endif()

foreach(subdir ${subdirs})
  add_hpx_pseudo_target(tests.performance.${subdir})
  add_subdirectory(${subdir})
  add_hpx_pseudo_dependencies(tests.performance tests.performance.${subdir})
endforeach()

