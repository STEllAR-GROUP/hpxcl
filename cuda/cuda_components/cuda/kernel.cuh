//  (C) Copyright 2013 Damond Howard
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(KERNEL_CUH)
#define KERNEL_CUH

//test functions
void cuda_malloc(void **devPtr, size_t size);
void cuda_test(long int* a);

//cuda device management functions
int get_devices();
void get_device_info();

//cuda kernel wrappers
float pi(int nthreads,int nblocks);
#endif //KERNEL_CUH
