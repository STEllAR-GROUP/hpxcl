# Copyright (c) 2015 Patrick Diehl
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(CUDA_NVCC_FLAGS "-use_fast_math; -gencode arch=compute_20,code=sm_20; 
-gencode arch=compute_20,code=sm_21; -gencode arch=compute_30,code=sm_30;
-gencode arch=compute_35,code=sm_35; -gencode arch=compute_50,code=sm_50;
-gencode arch=compute_52,code=sm_52; -gencode arch=compute_52,code=compute_52; -std=c++11" 
CACHE "NVCC_FLAGS" STRING FORCE) 

CUDA_ADD_EXECUTABLE(benchmarkCuda BenchmarkCUDA.cu)