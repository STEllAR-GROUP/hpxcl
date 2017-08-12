// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>

#include <hpxcl/opencl.hpp>

using namespace hpx::opencl;

static const char dgemm_src_str[] = 
"                                                                          \n"
"   __kernel void dgemm(const __global double *A, const __global double *B, __global double *C, const int m, const int n, const int k, const double alpha, const double beta)                       \n"
"   {                                                                      \n"
"       int ROW = get_global_id(1);                                 	   \n"
"       int COL = get_global_id(0);                                    	   \n"
"                                                                          \n"
"       if(ROW<(n) && COL<(m)){                                            \n"
"       	double sum = 0.0;                                              \n"
"       	for(int i = 0;i<k;i++)                                         \n"
"       		sum+=(alpha) * A[ROW * (k) + i] * B[i*(n)+COL];            \n"
"       	C[ROW*(n)+COL] = sum + (beta) * C[ROW*(n)+COL];                \n"
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

    // Get available OpenCL Devices.
    std::vector<device> devices = create_all_devices(CL_DEVICE_TYPE_ALL,
                                                     "OpenCL 1.1" ).get();

    // Check if any devices are available
    if(devices.size() < 1)
    {
        hpx::cerr << "No OpenCL devices found!" << hpx::endl;
        return hpx::finalize();
    }

    buffer_parameter_type m(1),n(1),k(1),i(1);
	buffer_parameter_type alpha(1), beta(1);

    //Initilizing the matrix dimensions
	m[0] = 2000;
	n[0] = 1000;
	k[0] = 200;

    // Create a device component from the first device found
    device cldevice = devices[0];

    buffer_data_type A(m*k), B (k*n), C(m*n);
	
	//initializing values of alpha and beta
	alpha = 1.0;
	beta = 0.0;

	printf (" Intializing matrix data \n\n");
	for (i = 0; i < (m*k); i++) {
		A[i] = (double)(i+1);
	}

	for (i = 0; i < (k*n); i++) {
		B[i] = (double)(-i-1);
	}

	for (i = 0; i < (m*n); i++) {
		C[i] = 0.0;
	}

	//creating buffers
	buffer ABuffer = cldevice.create_buffer(CL_MEM_READ_ONLY, m*k*sizeof( double ));
	buffer BBuffer = cldevice.create_buffer(CL_MEM_READ_ONLY, n*k*sizeof( double ));
	buffer CBuffer = cldevice.create_buffer(CL_MEM_READ_WRITE, m*n*sizeof( double ));
	buffer alphaBuffer = cldevice.create_buffer(CL_MEM_READ_ONLY, sizeof(double));
	buffer betaBuffer = cldevice.create_buffer(CL_MEM_READ_ONLY, sizeof(double));
	buffer mBuffer = cldevice.create_buffer(CL_MEM_READ_ONLY,sizeof(int));
	buffer nBuffer = cldevice.create_buffer(CL_MEM_READ_ONLY, sizeof(int));
	buffer kBuffer = cldevice.create_buffer(CL_MEM_READ_ONLY, sizeof(int));

	// Initialize a list of future events for asynchronous set_arg calls
    std::vector<shared_future<void>> set_arg_futures;
    std::vector<shared_future<void>> write_futures;

    // Create the hello_world device program
    program prog = cldevice.create_program_with_source(dgemm_src);

    //Build the program
    prog.build();

    //Write data to the buffers
    write_futures.push_back(ABuffer.enqueue_write(0, A, null));
    write_futures.push_back(BBuffer.enqueue_write(0, B, null));
    write_futures.push_back(CBuffer.enqueue_write(0, C, null));
    write_futures.push_back(alphaBuffer.enqueue_write(0, alpha, null));
    write_futures.push_back(betaBuffer.enqueue_write(0, beta, null));
    write_futures.push_back(mBuffer.enqueue_write(0, m, null));
    write_futures.push_back(nBuffer.enqueue_write(0, n, null));
    write_futures.push_back(kBuffer.enqueue_write(0, k, null));

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
    dim[0].size = m*k;
    dim[1].offset = 0;
    dim[1].size = n*k;

    hpx::future<void> kernel_future = dgemm_kernel.enqueue(dim); 

    // Start reading the buffer ( With kernel_future as dependency.
    //                            All hpxcl enqueue calls are nonblocking. )
    auto read_future = CBuffer.enqueue_read(0, m*n, kernel_future);

    // Wait for the data to arrive
    auto data = read_future.get();
    
    return 0;
}