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

static const char dgemm_src_str[] = 
"                                                                          \n"
"   __kernel void dgemm(__global double *A,__global double *B, __global double *C,__global int *m,__global int *n,__global int *k,__global double *alpha,__global double *beta)                       \n"
"   {                                                                      \n"
"       int ROW = get_global_id(1);                                 	   \n"
"       int COL = get_global_id(0);                                    	   \n"
"                                                                          \n"
"       if(ROW<(n[0]) && COL<(m[0])){                                            \n"
"       	double sum = 0.0;                                              \n"
"       	for(int i = 0;i<k[0];i++)                                         \n"
"       		sum+=(alpha[0]) * A[ROW * (k[0]) + i] * B[i*(n[0])+COL];            \n"
"       	C[ROW*(n[0])+COL] = sum + (beta[0]) * C[ROW*(n[0])+COL];                \n"
"       }                                                                  \n"
"                                                                          \n"
"   }                                                                      \n"
"                                                                          \n";

typedef hpx::serialization::serialize_buffer<char> buffer_type;
typedef hpx::serialization::serialize_buffer<double> buffer_data_type;
typedef hpx::serialization::serialize_buffer<int> buffer_parameter_type;

static buffer_type dgemm_src( dgemm_src_str,
                                    sizeof(dgemm_src_str),
                                    buffer_type::init_mode::reference );


int main(int argc, char* argv[])
{

	if (argc != 4) {
		std::cout << "Usage: " << argv[0] << " #m #n #k";
		exit(1);
	}

	int *m,*n,*k,i;

	//allocating memory for the vectors
	m = new int[1];
	n = new int[1];
	k = new int[1];

	//Initilizing the matrix dimensions
	m[0] = atoi(argv[1]);
	n[0] = atoi(argv[2]);
	k[0] = atoi(argv[3]);

	double time = 0;
	timer_start();

    // Get available OpenCL Devices.
    std::vector<device> devices = create_all_devices(CL_DEVICE_TYPE_ALL,
                                                     "OpenCL 1.1" ).get();

    // Check if any devices are available
    if(devices.size() < 1)
    {
        hpx::cerr << "No OpenCL devices found!" << hpx::endl;
        return hpx::finalize();
    }

    
	double *alpha, *beta;

	alpha = new double[1];
	beta = new double[1];

    // Create a device component from the first device found
    device cldevice = devices[0];

    double *A, *B, *C;
	
    A = new double[m[0]*k[0]];
    B = new double[k[0]*n[0]];
    C = new double[m[0]*n[0]];

	//initializing values of alpha and beta
	alpha[0] = 1.0;
	beta[0] = 0.0;

	time+=timer_stop();
	//printf (" Intializing matrix data \n\n");
	timer_start();
	for (i = 0; i < (m[0]*k[0]); i++) {
		A[i] = (double)(i+1);
	}

	for (i = 0; i < (k[0]*n[0]); i++) {
		B[i] = (double)(-i-1);
	}

	for (i = 0; i < (m[0]*n[0]); i++) {
		C[i] = 0.0;
	}

	//creating buffers
	buffer ABuffer = cldevice.create_buffer(CL_MEM_READ_ONLY, m[0]*k[0]*sizeof( double ));
	buffer BBuffer = cldevice.create_buffer(CL_MEM_READ_ONLY, n[0]*k[0]*sizeof( double ));
	buffer CBuffer = cldevice.create_buffer(CL_MEM_READ_WRITE, m[0]*n[0]*sizeof( double ));
	buffer alphaBuffer = cldevice.create_buffer(CL_MEM_READ_ONLY, sizeof(double));
	buffer betaBuffer = cldevice.create_buffer(CL_MEM_READ_ONLY, sizeof(double));
	buffer mBuffer = cldevice.create_buffer(CL_MEM_READ_ONLY,sizeof(int));
	buffer nBuffer = cldevice.create_buffer(CL_MEM_READ_ONLY, sizeof(int));
	buffer kBuffer = cldevice.create_buffer(CL_MEM_READ_ONLY, sizeof(int));

	// Initialize a list of future events for asynchronous set_arg calls
    std::vector<hpx::lcos::future<void>> set_arg_futures;
    std::vector<hpx::lcos::future<void>> write_futures;

    // Create the hello_world device program
    program prog = cldevice.create_program_with_source(dgemm_src);

    //Build the program
    prog.build();

    buffer_data_type A_serialized(
					A, m[0]*k[0],
					buffer_data_type::init_mode::reference);

    buffer_data_type B_serialized(
					B, k[0]*n[0],
					buffer_data_type::init_mode::reference);

    buffer_data_type C_serialized(
					C, m[0]*n[0],
					buffer_data_type::init_mode::reference);

	buffer_data_type alpha_serialized(
					alpha, 1,
					buffer_data_type::init_mode::reference);    

	buffer_data_type beta_serialized(
					beta, 1,
					buffer_data_type::init_mode::reference);

	buffer_parameter_type m_serialized(
					m, 1,
					buffer_parameter_type::init_mode::reference);

	buffer_parameter_type n_serialized(
					n, 1,
					buffer_parameter_type::init_mode::reference);

	buffer_parameter_type k_serialized(
					k, 1,
					buffer_parameter_type::init_mode::reference);

    //Write data to the buffers
    write_futures.push_back(ABuffer.enqueue_write(0, A_serialized));
    write_futures.push_back(BBuffer.enqueue_write(0, B_serialized));
    write_futures.push_back(CBuffer.enqueue_write(0, C_serialized));
    write_futures.push_back(alphaBuffer.enqueue_write(0, alpha_serialized));
    write_futures.push_back(betaBuffer.enqueue_write(0, beta_serialized));
    write_futures.push_back(mBuffer.enqueue_write(0, m_serialized));
    write_futures.push_back(nBuffer.enqueue_write(0, n_serialized));
    write_futures.push_back(kBuffer.enqueue_write(0, k_serialized));

    // wait for function calls to trigger
    hpx::wait_all( write_futures );

    //Creating the kernal
    kernel dgemm_kernel = prog.create_kernel("dgemm");

    //Set buffers as arguments
    set_arg_futures.push_back(dgemm_kernel.set_arg_async(0, ABuffer));
    set_arg_futures.push_back(dgemm_kernel.set_arg_async(1, BBuffer));
    set_arg_futures.push_back(dgemm_kernel.set_arg_async(2, CBuffer));
    set_arg_futures.push_back(dgemm_kernel.set_arg_async(3, mBuffer));
    set_arg_futures.push_back(dgemm_kernel.set_arg_async(4, nBuffer));
    set_arg_futures.push_back(dgemm_kernel.set_arg_async(5, kBuffer));
    set_arg_futures.push_back(dgemm_kernel.set_arg_async(6, alphaBuffer));
    set_arg_futures.push_back(dgemm_kernel.set_arg_async(7, betaBuffer));

    // wait for function calls to trigger
    hpx::wait_all( set_arg_futures );

    // Run the kernel
    hpx::opencl::work_size<2> dim;
    dim[0].offset = 0;
    dim[0].size = m[0];
    dim[1].offset = 0;
    dim[1].size = n[0];

    hpx::future<void> kernel_future = dgemm_kernel.enqueue(dim); 

    // Start reading the buffer ( With kernel_future as dependency.
    //                            All hpxcl enqueue calls are nonblocking. )
    auto read_future = CBuffer.enqueue_read(0, C_serialized, kernel_future);

    // Wait for the data to arrive
    auto data = read_future.get();

    //Printing the end timing result
    time+=timer_stop();
    std:: cout << time << std::endl;

    return 0;
}