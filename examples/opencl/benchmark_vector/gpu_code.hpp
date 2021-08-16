// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BENCHMARK_GPU_CODE_H_
#define BENCHMARK_GPU_CODE_H_

static const char gpu_code[] =
    "                                                                   \n"
    "   __kernel void logn(__global float* out,__global float* in)      \n"
    "   {                                                               \n"
    "       size_t tid = get_global_id(0);                              \n"
    "       out[tid] = log(in[tid]);                                    \n"
    "   }                                                               \n"
    "                                                                   \n"
    "   __kernel void expn(__global float* out,__global float* in)      \n"
    "   {                                                               \n"
    "       size_t tid = get_global_id(0);                              \n"
    "       out[tid] = exp(in[tid]);                                    \n"
    "   }                                                               \n"
    "                                                                   \n"
    "   __kernel void add(__global float* out,__global float* in1,      \n"
    "                                         __global float* in2)      \n"
    "   {                                                               \n"
    "       size_t tid = get_global_id(0);                              \n"
    "       out[tid] = in1[tid] + in2[tid];                             \n"
    "   }                                                               \n"
    "                                                                   \n"
    "   __kernel void dbl(__global float* out,__global float* in)       \n"
    "   {                                                               \n"
    "       size_t tid = get_global_id(0);                              \n"
    "       out[tid] = 2 * in[tid];                                     \n"
    "   }                                                               \n"
    "                                                                   \n"
    "   __kernel void mul(__global float* out,__global float* in1,      \n"
    "                                         __global float* in2)      \n"
    "   {                                                               \n"
    "       size_t tid = get_global_id(0);                              \n"
    "       out[tid] = in1[tid] * in2[tid];                             \n"
    "   }                                                               \n"
    "                                                                   \n";

#endif  // BENCHMARK_GPU_CODE_H_
