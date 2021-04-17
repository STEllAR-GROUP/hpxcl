# Copyright (c)       2016 Patrick Diehl
#					  2017 Madhavan Seshadri
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(subdirs "")

if(HPXCL_WITH_NAIVE_OPENCL_BENCHMARK)
  set(subdirs
    ${subdirs} opencl)
    list(append HPXCL_TARGETS benchmark/opencl)
endif()

if(HPXCL_WITH_CUDA)
  set(subdirs
    ${subdirs} cuda)
    list(append HPXCL_TARGETS benchmark/cuda)
endif()


foreach(subdir ${subdirs})
    add_hpx_pseudo_target(benchmark.${subdir})
    add_subdirectory(${subdir})
    add_hpx_pseudo_dependencies(benchmark benchmark.${subdir})
endforeach()
