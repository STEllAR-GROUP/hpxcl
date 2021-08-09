// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BENCHMARK_DIRECTCL_HPP_
#define BENCHMARK_DIRECTCL_HPP_

#include "gpu_code.hpp"

#include <vector>
#include <cmath>
#include <memory>

#include <CL/cl.h>

static cl_context directcl_context;
static cl_command_queue directcl_command_queue;
static cl_program directcl_program;
static cl_kernel directcl_exp_kernel;
static cl_kernel directcl_log_kernel;
static cl_kernel directcl_add_kernel;
static cl_kernel directcl_mul_kernel;
static cl_kernel directcl_dbl_kernel;
static cl_mem directcl_buffer_a;
static cl_mem directcl_buffer_b;
static cl_mem directcl_buffer_c;
static cl_mem directcl_buffer_m;
static cl_mem directcl_buffer_n;
static cl_mem directcl_buffer_o;
static cl_mem directcl_buffer_p;
static cl_mem directcl_buffer_z;

#define directcl_check(ret)                                               \
  {                                                                       \
    if ((ret) != CL_SUCCESS) {                                            \
      hpx::cout << "directcl.hpp:" << __LINE__ << ": CL ERROR: " << (ret) \
                << hpx::endl;                                             \
      exit(1);                                                            \
    }                                                                     \
  }

/*static void directcl_check(cl_int ret)
{

    if(ret != CL_SUCCESS){
        hpx::cout << "CL ERROR: " << ret << hpx::endl;
        exit(1);
    }

}*/

static cl_device_id directcl_choose_device() {
  cl_int ret;

  // get number of platform ids
  cl_uint num_platforms;
  ret = clGetPlatformIDs(0, NULL, &num_platforms);
  directcl_check(ret);

  // get platform ids
  std::vector<cl_platform_id> platforms(num_platforms);
  ret = clGetPlatformIDs(num_platforms, platforms.data(), NULL);
  directcl_check(ret);

  /*
      // Print Platforms
      hpx::cout << "Platforms:" << hpx::endl;
      for(cl_uint i = 0; i < num_platforms; i++)
      {
          char platformName[100];
          char platformVendor[100];

          ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 100,
                                  platformName, NULL);
          directcl_check(ret);
          ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 100,
                                  platformVendor, NULL);
          directcl_check(ret);

          hpx::cout << i << ": " << platformName << " (" << platformVendor <<
     ")"
                    << hpx::endl;
      }

      // Lets you choose a platform
      cl_uint platform_num;
      hpx::cout << "Choose platform: " << hpx::endl;
      std::cin >> platform_num;
      if(platform_num < 0 || platform_num >= num_platforms)
          exit(0);
  */

  // Ensure that we found a platforms
  if (num_platforms < 1) {
    hpx::cout << "No OpenCL platforms found!" << hpx::endl;
    exit(1);
  }

  // Select the platform
  cl_uint num_devices = 0;
  cl_platform_id platform = 0;
  for (auto& current_platform : platforms) {
    // get number of device ids
    ret = clGetDeviceIDs(current_platform, CL_DEVICE_TYPE_GPU, 0, NULL,
                         &num_devices);
    if (ret == CL_DEVICE_NOT_FOUND) continue;
    directcl_check(ret);

    // Print platform name
    hpx::cout << "Platform:" << hpx::endl;
    {
      char platformName[100];
      char platformVendor[100];

      ret = clGetPlatformInfo(current_platform, CL_PLATFORM_NAME, 100,
                              platformName, NULL);
      directcl_check(ret);
      ret = clGetPlatformInfo(current_platform, CL_PLATFORM_VENDOR, 100,
                              platformVendor, NULL);
      directcl_check(ret);

      hpx::cout << "    " << platformName << " (" << platformVendor << ")"
                << hpx::endl;
    }

    platform = current_platform;
    break;
  }

  // Ensure that we found a platforms
  if (num_devices < 1) {
    hpx::cout << "No OpenCL devices found!" << hpx::endl;
    exit(1);
  }

  // get device ids
  std::vector<cl_device_id> devices(num_devices);
  ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices,
                       devices.data(), NULL);

  // Print devices
  hpx::cout << "Device:" << hpx::endl;
  {
    char deviceName[100];

    ret = clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 100, deviceName, NULL);
    directcl_check(ret);

    hpx::cout << "    " << deviceName << hpx::endl;
  }

  return devices[0];
}

static void directcl_initialize(size_t vector_size) {
  cl_device_id device_id = directcl_choose_device();

  cl_int err;

  // Create context
  directcl_context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  directcl_check(err);

  // Create command queue
  directcl_command_queue =
      clCreateCommandQueue(directcl_context, device_id, 0, &err);
  directcl_check(err);

  // Create program
  const char* gpu_code_ptr = gpu_code;
  directcl_program =
      clCreateProgramWithSource(directcl_context, 1, &gpu_code_ptr, NULL, &err);
  directcl_check(err);

  // Build program
  err = clBuildProgram(directcl_program, 1, &device_id, NULL, NULL, NULL);

  // Create kernels
  directcl_log_kernel = clCreateKernel(directcl_program, "logn", &err);
  directcl_check(err);
  directcl_exp_kernel = clCreateKernel(directcl_program, "expn", &err);
  directcl_check(err);
  directcl_mul_kernel = clCreateKernel(directcl_program, "mul", &err);
  directcl_check(err);
  directcl_add_kernel = clCreateKernel(directcl_program, "add", &err);
  directcl_check(err);
  directcl_dbl_kernel = clCreateKernel(directcl_program, "dbl", &err);
  directcl_check(err);

  // Create buffers
  directcl_buffer_a = clCreateBuffer(directcl_context, CL_MEM_READ_ONLY,
                                     vector_size * sizeof(float), NULL, &err);
  directcl_check(err);
  directcl_buffer_b = clCreateBuffer(directcl_context, CL_MEM_READ_ONLY,
                                     vector_size * sizeof(float), NULL, &err);
  directcl_check(err);
  directcl_buffer_c = clCreateBuffer(directcl_context, CL_MEM_READ_ONLY,
                                     vector_size * sizeof(float), NULL, &err);
  directcl_check(err);
  directcl_buffer_m = clCreateBuffer(directcl_context, CL_MEM_READ_WRITE,
                                     vector_size * sizeof(float), NULL, &err);
  directcl_check(err);
  directcl_buffer_n = clCreateBuffer(directcl_context, CL_MEM_READ_WRITE,
                                     vector_size * sizeof(float), NULL, &err);
  directcl_check(err);
  directcl_buffer_o = clCreateBuffer(directcl_context, CL_MEM_READ_WRITE,
                                     vector_size * sizeof(float), NULL, &err);
  directcl_check(err);
  directcl_buffer_p = clCreateBuffer(directcl_context, CL_MEM_READ_WRITE,
                                     vector_size * sizeof(float), NULL, &err);
  directcl_check(err);
  directcl_buffer_z = clCreateBuffer(directcl_context, CL_MEM_WRITE_ONLY,
                                     vector_size * sizeof(float), NULL, &err);
  directcl_check(err);

  // set kernel args for exp
  err = clSetKernelArg(directcl_exp_kernel, 0, sizeof(cl_mem),
                       &directcl_buffer_m);
  directcl_check(err);
  err = clSetKernelArg(directcl_exp_kernel, 1, sizeof(cl_mem),
                       &directcl_buffer_b);
  directcl_check(err);

  // set kernel args for add
  err = clSetKernelArg(directcl_add_kernel, 0, sizeof(cl_mem),
                       &directcl_buffer_n);
  directcl_check(err);
  err = clSetKernelArg(directcl_add_kernel, 1, sizeof(cl_mem),
                       &directcl_buffer_a);
  directcl_check(err);
  err = clSetKernelArg(directcl_add_kernel, 2, sizeof(cl_mem),
                       &directcl_buffer_m);
  directcl_check(err);

  // set kernel args for dbl
  err = clSetKernelArg(directcl_dbl_kernel, 0, sizeof(cl_mem),
                       &directcl_buffer_o);
  directcl_check(err);
  err = clSetKernelArg(directcl_dbl_kernel, 1, sizeof(cl_mem),
                       &directcl_buffer_c);
  directcl_check(err);

  // set kernel args for mul
  err = clSetKernelArg(directcl_mul_kernel, 0, sizeof(cl_mem),
                       &directcl_buffer_p);
  directcl_check(err);
  err = clSetKernelArg(directcl_mul_kernel, 1, sizeof(cl_mem),
                       &directcl_buffer_n);
  directcl_check(err);
  err = clSetKernelArg(directcl_mul_kernel, 2, sizeof(cl_mem),
                       &directcl_buffer_o);
  directcl_check(err);

  // set kernel args for log
  err = clSetKernelArg(directcl_log_kernel, 0, sizeof(cl_mem),
                       &directcl_buffer_z);
  directcl_check(err);
  err = clSetKernelArg(directcl_log_kernel, 1, sizeof(cl_mem),
                       &directcl_buffer_p);
  directcl_check(err);
}

static std::shared_ptr<std::vector<float>> directcl_calculate(
    hpx::serialization::serialize_buffer<float> a,
    hpx::serialization::serialize_buffer<float> b,
    hpx::serialization::serialize_buffer<float> c, double* t_nonblock,
    double* t_total) {
  // do nothing if matrices are wrong
  if (a.size() != b.size() || b.size() != c.size()) {
    return std::shared_ptr<std::vector<float>>();
  }

  // initialize error test
  cl_int err;

  // copy data to gpu
  err = clEnqueueWriteBuffer(directcl_command_queue, directcl_buffer_a,
                             CL_FALSE, 0, a.size() * sizeof(float), a.data(), 0,
                             NULL, NULL);
  directcl_check(err);
  err = clEnqueueWriteBuffer(directcl_command_queue, directcl_buffer_b,
                             CL_FALSE, 0, a.size() * sizeof(float), b.data(), 0,
                             NULL, NULL);
  directcl_check(err);
  err = clEnqueueWriteBuffer(directcl_command_queue, directcl_buffer_c,
                             CL_FALSE, 0, a.size() * sizeof(float), c.data(), 0,
                             NULL, NULL);
  directcl_check(err);

  // wait for writes to finish
  err = clFinish(directcl_command_queue);
  directcl_check(err);

  // start timer
  timer_start();

  // run kernels
  size_t size = a.size();
  err = clEnqueueNDRangeKernel(directcl_command_queue, directcl_exp_kernel, 1,
                               NULL, &size, NULL, 0, NULL, NULL);
  directcl_check(err);
  err = clEnqueueNDRangeKernel(directcl_command_queue, directcl_add_kernel, 1,
                               NULL, &size, NULL, 0, NULL, NULL);
  directcl_check(err);
  err = clEnqueueNDRangeKernel(directcl_command_queue, directcl_dbl_kernel, 1,
                               NULL, &size, NULL, 0, NULL, NULL);
  directcl_check(err);
  err = clEnqueueNDRangeKernel(directcl_command_queue, directcl_mul_kernel, 1,
                               NULL, &size, NULL, 0, NULL, NULL);
  directcl_check(err);
  err = clEnqueueNDRangeKernel(directcl_command_queue, directcl_log_kernel, 1,
                               NULL, &size, NULL, 0, NULL, NULL);
  directcl_check(err);

  // get time of nonblocking calls
  *t_nonblock = timer_stop();

  // finish
  err = clFinish(directcl_command_queue);
  directcl_check(err);

  // get time of total calculation
  *t_total = timer_stop();

  // allocate the result buffer
  std::shared_ptr<std::vector<float>> res(new std::vector<float>(a.size()));

  // read into result buffer
  err = clEnqueueReadBuffer(directcl_command_queue, directcl_buffer_z, CL_FALSE,
                            0, a.size() * sizeof(float), res->data(), 0, NULL,
                            NULL);
  directcl_check(err);

  // finish
  err = clFinish(directcl_command_queue);
  directcl_check(err);

  return res;
}

static void directcl_shutdown() {
  cl_int err;

  // release buffers
  err = clReleaseMemObject(directcl_buffer_a);
  directcl_check(err);
  err = clReleaseMemObject(directcl_buffer_b);
  directcl_check(err);
  err = clReleaseMemObject(directcl_buffer_c);
  directcl_check(err);
  err = clReleaseMemObject(directcl_buffer_m);
  directcl_check(err);
  err = clReleaseMemObject(directcl_buffer_n);
  directcl_check(err);
  err = clReleaseMemObject(directcl_buffer_o);
  directcl_check(err);
  err = clReleaseMemObject(directcl_buffer_p);
  directcl_check(err);
  err = clReleaseMemObject(directcl_buffer_z);
  directcl_check(err);

  // release kernels
  err = clReleaseKernel(directcl_dbl_kernel);
  directcl_check(err);
  err = clReleaseKernel(directcl_add_kernel);
  directcl_check(err);
  err = clReleaseKernel(directcl_mul_kernel);
  directcl_check(err);
  err = clReleaseKernel(directcl_exp_kernel);
  directcl_check(err);
  err = clReleaseKernel(directcl_log_kernel);
  directcl_check(err);

  // release program
  err = clReleaseProgram(directcl_program);
  directcl_check(err);

  // release command queue
  err = clReleaseCommandQueue(directcl_command_queue);
  directcl_check(err);

  // release context
  err = clReleaseContext(directcl_context);
  directcl_check(err);
}

#endif  // BENCHMARK_DIRECTCL_HPP_
