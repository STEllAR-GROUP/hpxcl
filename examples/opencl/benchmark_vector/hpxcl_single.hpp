// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BENCHMARK_HPXCL_SINGLE_HPP_
#define BENCHMARK_HPXCL_SINGLE_HPP_

#include <hpxcl/opencl.hpp>
#include "timer.hpp"

using namespace hpx::opencl;
using hpx::lcos::shared_future;

static device   hpxcl_single_device;
static buffer   hpxcl_single_buffer_a;
static buffer   hpxcl_single_buffer_b;
static buffer   hpxcl_single_buffer_c;
static buffer   hpxcl_single_buffer_m;
static buffer   hpxcl_single_buffer_n;
static buffer   hpxcl_single_buffer_o;
static buffer   hpxcl_single_buffer_p;
static buffer   hpxcl_single_buffer_z;
static program  hpxcl_single_program;
static kernel   hpxcl_single_log_kernel;
static kernel   hpxcl_single_exp_kernel;
static kernel   hpxcl_single_mul_kernel;
static kernel   hpxcl_single_add_kernel;
static kernel   hpxcl_single_dbl_kernel;


static void hpxcl_single_initialize( hpx::naming::id_type node_id,
                                     size_t vector_size)
{

    // Query all devices on local node
    std::vector<device> devices = create_devices( node_id,
                                                  CL_DEVICE_TYPE_GPU,
                                                  "OpenCL 1.1" ).get();

/*
    // print devices
    hpx::cout << "Devices:" << hpx::endl;
    for(cl_uint i = 0; i < devices.size(); i++)
    {

        device cldevice = devices[i];

        // Query name
        std::string device_name =
            cldevice.get_device_info<CL_DEVICE_NAME>().get();
        std::string device_vendor =
            cldevice.get_device_info<CL_DEVICE_VENDOR>().get();

        hpx::cout << i << ": " << device_name << " (" << device_vendor << ")"
                  << hpx::endl;

    }

    // Lets you choose a device
    size_t device_num;
    hpx::cout << "Choose device: " << hpx::endl;
    std::cin >> device_num;
    if(device_num >= devices.size())
        exit(0);

    // Select a device
    hpxcl_single_device = devices[device_num];
*/

    size_t device_id = 0;
    // print device
    hpx::cout << "Device:" << hpx::endl;
    {

        device cldevice = devices[device_id];

        // Query name
        std::string device_name =
            cldevice.get_device_info<CL_DEVICE_NAME>().get();
        std::string device_vendor =
            cldevice.get_device_info<CL_DEVICE_VENDOR>().get();

        hpx::cout << "    " << device_name << " (" << device_vendor << ")"
                  << hpx::endl;

    }

    // Select a device
    hpxcl_single_device = devices[device_id];

    // Create program
    typedef hpx::serialization::serialize_buffer<char> buffer_type;
    buffer_type gpu_code_buffer( gpu_code, sizeof(gpu_code),
                                 buffer_type::init_mode::reference );
    hpxcl_single_program =
        hpxcl_single_device.create_program_with_source( gpu_code_buffer );

    // Build program
    hpxcl_single_program.build();

    // Create kernels
    hpxcl_single_log_kernel = hpxcl_single_program.create_kernel("logn");
    hpxcl_single_exp_kernel = hpxcl_single_program.create_kernel("expn");
    hpxcl_single_mul_kernel = hpxcl_single_program.create_kernel("mul");
    hpxcl_single_add_kernel = hpxcl_single_program.create_kernel("add");
    hpxcl_single_dbl_kernel = hpxcl_single_program.create_kernel("dbl");

    // Generate buffers
    hpxcl_single_buffer_a = hpxcl_single_device.create_buffer(
                                    CL_MEM_READ_ONLY,
                                    vector_size * sizeof(float));
    hpxcl_single_buffer_b = hpxcl_single_device.create_buffer(
                                    CL_MEM_READ_ONLY,
                                    vector_size * sizeof(float));
    hpxcl_single_buffer_c = hpxcl_single_device.create_buffer(
                                    CL_MEM_READ_ONLY,
                                    vector_size * sizeof(float));
    hpxcl_single_buffer_m = hpxcl_single_device.create_buffer(
                                    CL_MEM_READ_WRITE,
                                    vector_size * sizeof(float));
    hpxcl_single_buffer_n = hpxcl_single_device.create_buffer(
                                    CL_MEM_READ_WRITE,
                                    vector_size * sizeof(float));
    hpxcl_single_buffer_o = hpxcl_single_device.create_buffer(
                                    CL_MEM_READ_WRITE,
                                    vector_size * sizeof(float));
    hpxcl_single_buffer_p = hpxcl_single_device.create_buffer(
                                    CL_MEM_READ_WRITE,
                                    vector_size * sizeof(float));
    hpxcl_single_buffer_z = hpxcl_single_device.create_buffer(
                                    CL_MEM_WRITE_ONLY,
                                    vector_size * sizeof(float));

    // Initialize a list of future events for asynchronous set_arg calls
    std::vector<shared_future<void>> set_arg_futures;

    // set kernel args for exp
    set_arg_futures.push_back(
        hpxcl_single_exp_kernel.set_arg_async(0, hpxcl_single_buffer_m));
    set_arg_futures.push_back(
        hpxcl_single_exp_kernel.set_arg_async(1, hpxcl_single_buffer_b));

    // set kernel args for add
    set_arg_futures.push_back(
        hpxcl_single_add_kernel.set_arg_async(0, hpxcl_single_buffer_n));
    set_arg_futures.push_back(
        hpxcl_single_add_kernel.set_arg_async(1, hpxcl_single_buffer_a));
    set_arg_futures.push_back(
        hpxcl_single_add_kernel.set_arg_async(2, hpxcl_single_buffer_m));

    // set kernel args for dbl
    set_arg_futures.push_back(
        hpxcl_single_dbl_kernel.set_arg_async(0, hpxcl_single_buffer_o));
    set_arg_futures.push_back(
        hpxcl_single_dbl_kernel.set_arg_async(1, hpxcl_single_buffer_c));

    // set kernel args for mul
    set_arg_futures.push_back(
        hpxcl_single_mul_kernel.set_arg_async(0, hpxcl_single_buffer_p));
    set_arg_futures.push_back(
        hpxcl_single_mul_kernel.set_arg_async(1, hpxcl_single_buffer_n));
    set_arg_futures.push_back(
        hpxcl_single_mul_kernel.set_arg_async(2, hpxcl_single_buffer_o));

    // set kernel args for log
    set_arg_futures.push_back(
        hpxcl_single_log_kernel.set_arg_async(0, hpxcl_single_buffer_z));
    set_arg_futures.push_back(
        hpxcl_single_log_kernel.set_arg_async(1, hpxcl_single_buffer_p));

    // wait for function calls to trigger
    hpx::wait_all( set_arg_futures );


}

static hpx::serialization::serialize_buffer<float>
hpxcl_single_calculate(hpx::serialization::serialize_buffer<float> a,
                       hpx::serialization::serialize_buffer<float> b,
                       hpx::serialization::serialize_buffer<float> c,
                       double* t_nonblock,
                       double* t_finish)
{
    // do nothing if matrices are wrong
    if(a.size() != b.size() || b.size() != c.size())
    {
        return hpx::serialization::serialize_buffer<float>();
    }

    size_t size = a.size();

    // copy data to gpu
    auto write_a_event = hpxcl_single_buffer_a.enqueue_write( 0, a );
    auto write_b_event = hpxcl_single_buffer_b.enqueue_write( 0, b );
    auto write_c_event = hpxcl_single_buffer_c.enqueue_write( 0, c );

    // wait for write to finish
    write_a_event.wait();
    write_b_event.wait();
    write_c_event.wait();

    // start time measurement
    timer_start();

    // set work dimensions
    work_size<1> dim;
    dim[0].offset = 0;
    dim[0].size = size;

    // run exp kernel
    auto kernel_exp_event = hpxcl_single_exp_kernel.enqueue(dim, write_b_event);

    // run add kernel
    auto kernel_add_event = hpxcl_single_add_kernel.enqueue( dim,
                                                             kernel_exp_event,
                                                             write_a_event );

    // run dbl kernel
    auto kernel_dbl_event = hpxcl_single_dbl_kernel.enqueue( dim,
                                                             write_c_event );

    // run mul kernel
    auto kernel_mul_event = hpxcl_single_mul_kernel.enqueue( dim,
                                                             kernel_add_event,
                                                             kernel_dbl_event );

    // run log kernel
    auto kernel_log_event = hpxcl_single_log_kernel.enqueue( dim,
                                                             kernel_mul_event);

    ////////// UNTIL HERE ALL CALLS WERE NON-BLOCKING /////////////////////////

    // get time of non-blocking calls
    *t_nonblock = timer_stop();

    // wait for the end of the execution
    kernel_log_event.wait();

    // get total time of execution
    *t_finish = timer_stop();

    // enqueue result read
    typedef hpx::serialization::serialize_buffer<float> buffer_type;
    buffer_type result_buffer ( size );
    auto read_event =
        hpxcl_single_buffer_z.enqueue_read( 0, result_buffer,
                                            kernel_log_event );

    // wait for calculation to complete and return data
    return read_event.get();

}

static void hpxcl_single_shutdown()
{

    // release buffers
    hpxcl_single_buffer_a = buffer();
    hpxcl_single_buffer_b = buffer();
    hpxcl_single_buffer_c = buffer();
    hpxcl_single_buffer_m = buffer();
    hpxcl_single_buffer_n = buffer();
    hpxcl_single_buffer_o = buffer();
    hpxcl_single_buffer_p = buffer();
    hpxcl_single_buffer_z = buffer();

    // release kernels
    hpxcl_single_dbl_kernel = kernel();
    hpxcl_single_add_kernel = kernel();
    hpxcl_single_mul_kernel = kernel();
    hpxcl_single_exp_kernel = kernel();
    hpxcl_single_log_kernel = kernel();

    // release program
    hpxcl_single_program = program();

    // delete device
    hpxcl_single_device = device();

}





















#endif //BENCHMARK_HPXCL_SINGLE_HPP_

