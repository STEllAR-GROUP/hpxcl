// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>

#include "examples/opencl/benchmark_vector/timer.hpp"
#include <hpxcl/opencl.hpp>

using namespace hpx::opencl;

static const char smvp_src_str[] =
    "                                                                          "
    "					   \n"
    "__kernel void smvp(__global double *A_data,__global int *A_indices, "
    "__global int *A_pointers, \n"
    "__global double *B, __global double *C, __global int *m, __global int *n, "
    "__global int *count,\n"
    "__global double *alpha)						"
    "									"
    "		   \n"
    "{									"
    "									"
    "					   \n"
    "	int ROW = get_global_id(0);					"
    "									"
    "		   \n"
    "									"
    "									"
    "					   \n"
    "	if(ROW<m[0]){							"
    "									"
    "			   \n"
    "		int start = A_pointers[ROW];				"
    "									"
    "	   \n"
    "		int end = (start==m[0]-1)?(count[0]):A_pointers[ROW+1];	"
    "							   \n"
    "									"
    "									"
    "					   \n"
    "		double sum = 0;						"
    "									"
    "			   \n"
    "		for(int i = start;i<end;i++)				"
    "									"
    "	   \n"
    "		{							"
    "									"
    "					   \n"
    "			int index = A_indices[i];			"
    "									"
    "		   \n"
    "			sum += (alpha[0]) * A_data[i] * B[index];	"
    "									   \n"
    "		}							"
    "									"
    "					   \n"
    "		C[ROW] = sum;						"
    "									"
    "			   \n"
    "	}								"
    "									"
    "					   \n"
    "}									"
    "									"
    "					   \n"
    "                                                                          "
    "					   \n";

typedef hpx::serialization::serialize_buffer<char> buffer_type;
typedef hpx::serialization::serialize_buffer<double> buffer_data_type;
typedef hpx::serialization::serialize_buffer<int> buffer_parameter_type;

static buffer_type smvp_src(smvp_src_str, sizeof(smvp_src_str),
                            buffer_type::init_mode::reference);

//###########################################################################
// Main
//###########################################################################
int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " #m #n";
    exit(1);
  }

  int *m, *n, i;

  // allocating memory for the vectors
  m = new int[1];
  n = new int[1];

  m[0] = atoi(argv[1]);
  n[0] = atoi(argv[2]);

  double time = 0;
  timer_start();

  // Get available OpenCL Devices.
  std::vector<device> devices =
      create_all_devices(CL_DEVICE_TYPE_ALL, "OpenCL 1.1").get();

  // Check if any devices are available
  if (devices.size() < 1) {
    hpx::cerr << "No OpenCL devices found!" << hpx::endl;
    return hpx::finalize();
  }

  double *alpha;
  int *count;

  alpha = new double[1];
  count = new int[1];

  // Create a device component from the first device found
  device cldevice = devices[0];

  // Create the hello_world device program
  program prog = cldevice.create_program_with_source(smvp_src);

  // Build the program
  auto program_future = prog.build_async();

  double *A, *B, *C;

  double *A_data;
  int *A_indices, *A_pointers;

  A = new double[m[0] * n[0]];
  B = new double[n[0]];
  C = new double[m[0]];

  // initializing values of alpha and beta
  alpha[0] = 1.0;
  count[0] = 0;

  // Input can be anything sparse
  for (i = 0; i < (m[0] * n[0]); i++) {
    if ((i % n[0]) == 0) {
      A[i] = (double)(i + 1);
      count[0]++;
    }
  }

  A_data = new double[count[0]];
  A_indices = new int[count[0]];
  A_pointers = new int[m[0]];

  for (i = 0; i < (1 * n[0]); i++) {
    B[i] = (double)(-i - 1);
  }

  for (i = 0; i < (m[0] * 1); i++) {
    C[i] = 0.0;
  }

  // Counters for compression
  int data_counter = 0;
  int index_counter = 0;
  int pointer_counter = -1;

  // Compressing Matrix A
  for (i = 0; i < (m[0] * n[0]); i++) {
    if (A[i] != 0) {
      A_data[data_counter++] = A[i];
      if (((int)i / n[0]) != pointer_counter)
        A_pointers[++pointer_counter] = index_counter;
      A_indices[index_counter++] = (i % n[0]);
    }
  }

  // creating buffers
  buffer ADataBuffer =
      cldevice.create_buffer(CL_MEM_READ_ONLY, (count[0]) * sizeof(double));
  buffer AIndexBuffer =
      cldevice.create_buffer(CL_MEM_READ_ONLY, (count[0]) * sizeof(int));
  buffer APointerBuffer =
      cldevice.create_buffer(CL_MEM_READ_ONLY, m[0] * sizeof(int));

  buffer BBuffer =
      cldevice.create_buffer(CL_MEM_READ_ONLY, n[0] * sizeof(double));
  buffer CBuffer =
      cldevice.create_buffer(CL_MEM_READ_WRITE, m[0] * sizeof(double));
  buffer alphaBuffer = cldevice.create_buffer(CL_MEM_READ_ONLY, sizeof(double));
  buffer countBuffer = cldevice.create_buffer(CL_MEM_READ_ONLY, sizeof(int));
  buffer mBuffer = cldevice.create_buffer(CL_MEM_READ_ONLY, sizeof(int));
  buffer nBuffer = cldevice.create_buffer(CL_MEM_READ_ONLY, sizeof(int));

  // Initialize a list of future events for asynchronous set_arg calls
  std::vector<hpx::lcos::future<void>> set_arg_futures;
  std::vector<hpx::lcos::future<void>> write_futures;

  buffer_data_type AData_serialized(A_data, (*count),
                                    buffer_data_type::init_mode::reference);

  buffer_parameter_type AIndex_serialized(
      A_indices, (*count), buffer_parameter_type::init_mode::reference);

  buffer_parameter_type APointer_serialized(
      A_pointers, m[0], buffer_parameter_type::init_mode::reference);

  buffer_data_type B_serialized(B, n[0],
                                buffer_data_type::init_mode::reference);

  buffer_data_type C_serialized(C, m[0],
                                buffer_data_type::init_mode::reference);

  buffer_data_type alpha_serialized(alpha, 1,
                                    buffer_data_type::init_mode::reference);

  buffer_parameter_type count_serialized(
      count, 1, buffer_parameter_type::init_mode::reference);

  buffer_parameter_type m_serialized(
      m, 1, buffer_parameter_type::init_mode::reference);

  buffer_parameter_type n_serialized(
      n, 1, buffer_parameter_type::init_mode::reference);

  // Write data to the buffers
  write_futures.push_back(ADataBuffer.enqueue_write(0, AData_serialized));
  write_futures.push_back(AIndexBuffer.enqueue_write(0, AIndex_serialized));
  write_futures.push_back(APointerBuffer.enqueue_write(0, APointer_serialized));

  write_futures.push_back(BBuffer.enqueue_write(0, B_serialized));
  write_futures.push_back(CBuffer.enqueue_write(0, C_serialized));
  write_futures.push_back(alphaBuffer.enqueue_write(0, alpha_serialized));
  write_futures.push_back(mBuffer.enqueue_write(0, m_serialized));
  write_futures.push_back(nBuffer.enqueue_write(0, n_serialized));
  write_futures.push_back(countBuffer.enqueue_write(0, count_serialized));

  // wait for function calls to trigger
  hpx::wait_all(write_futures);

  hpx::wait_all(program_future);

  // Creating the kernal
  kernel smvp_kernel = prog.create_kernel("smvp");

  // Set buffers as arguments
  set_arg_futures.push_back(smvp_kernel.set_arg_async(0, ADataBuffer));
  set_arg_futures.push_back(smvp_kernel.set_arg_async(1, AIndexBuffer));
  set_arg_futures.push_back(smvp_kernel.set_arg_async(2, APointerBuffer));
  set_arg_futures.push_back(smvp_kernel.set_arg_async(3, BBuffer));
  set_arg_futures.push_back(smvp_kernel.set_arg_async(4, CBuffer));
  set_arg_futures.push_back(smvp_kernel.set_arg_async(5, mBuffer));
  set_arg_futures.push_back(smvp_kernel.set_arg_async(6, nBuffer));
  set_arg_futures.push_back(smvp_kernel.set_arg_async(7, countBuffer));
  set_arg_futures.push_back(smvp_kernel.set_arg_async(8, alphaBuffer));

  // wait for function calls to trigger
  hpx::wait_all(set_arg_futures);

  // Run the kernel
  hpx::opencl::work_size<1> dim;
  dim[0].offset = 0;
  dim[0].size = (int)(std::pow(2, std::ceil(std::log(m[0]) / std::log(2))));
  dim[0].local_size = 32;

  hpx::future<void> kernel_future = smvp_kernel.enqueue(dim);

  // Start reading the buffer ( With kernel_future as dependency.
  //                            All hpxcl enqueue calls are nonblocking. )
  auto read_future = CBuffer.enqueue_read(0, C_serialized, kernel_future);

  // Wait for the data to arrive
  auto data = read_future.get();

  // Printing the end timing result
  time += timer_stop();
  std::cout << time << std::endl;

  return 0;
}