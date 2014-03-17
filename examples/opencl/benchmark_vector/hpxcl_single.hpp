// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BENCHMARK_HPXCL_SINGLE_HPP__
#define BENCHMARK_HPXCL_SINGLE_HPP__

#include "../../../opencl.hpp"
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
    std::vector<device> devices = get_devices( node_id, 
                                               CL_DEVICE_TYPE_ALL,
                                               1.1f ).get();

    // print devices
    hpx::cout << "Devices:" << hpx::endl;
    for(cl_uint i = 0; i < devices.size(); i++)
    {
        
        device cldevice = devices[i];

        // Query name
        std::string device_name = device::device_info_to_string(
                                    cldevice.get_device_info(CL_DEVICE_NAME));
        std::string device_vendor = device::device_info_to_string(
                                    cldevice.get_device_info(CL_DEVICE_VENDOR));

        hpx::cout << i << ": " << device_name << " (" << device_vendor << ")"
                  << hpx::endl;

    }

    // Lets you choose a device
    size_t device_num;
    hpx::cout << "Choose device: " << hpx::endl;
    std::cin >> device_num;
    if(device_num < 0 || device_num >= devices.size())
        exit(0);

    // Select a device
    hpxcl_single_device = devices[device_num];

    // Create program
    hpxcl_single_program = hpxcl_single_device.create_program_with_source(
                                                                    gpu_code);

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
                                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                    vector_size * sizeof(float));
    hpxcl_single_buffer_b = hpxcl_single_device.create_buffer(
                                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                    vector_size * sizeof(float));
    hpxcl_single_buffer_c = hpxcl_single_device.create_buffer(
                                    CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                    vector_size * sizeof(float));
    hpxcl_single_buffer_m = hpxcl_single_device.create_buffer(
                                    CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                    vector_size * sizeof(float));
    hpxcl_single_buffer_n = hpxcl_single_device.create_buffer(
                                    CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                    vector_size * sizeof(float));
    hpxcl_single_buffer_o = hpxcl_single_device.create_buffer(
                                    CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                    vector_size * sizeof(float));
    hpxcl_single_buffer_p = hpxcl_single_device.create_buffer(
                                    CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                    vector_size * sizeof(float));
    hpxcl_single_buffer_z = hpxcl_single_device.create_buffer(
                                    CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY,
                                    vector_size * sizeof(float));

    // set kernel args for exp
    hpxcl_single_exp_kernel.set_arg(0, hpxcl_single_buffer_m);
    hpxcl_single_exp_kernel.set_arg(1, hpxcl_single_buffer_b);
    
    // set kernel args for add
    hpxcl_single_add_kernel.set_arg(0, hpxcl_single_buffer_n);
    hpxcl_single_add_kernel.set_arg(1, hpxcl_single_buffer_a);
    hpxcl_single_add_kernel.set_arg(2, hpxcl_single_buffer_m);

    // set kernel args for dbl
    hpxcl_single_dbl_kernel.set_arg(0, hpxcl_single_buffer_o);
    hpxcl_single_dbl_kernel.set_arg(1, hpxcl_single_buffer_c);
    
    // set kernel args for mul
    hpxcl_single_mul_kernel.set_arg(0, hpxcl_single_buffer_p);
    hpxcl_single_mul_kernel.set_arg(1, hpxcl_single_buffer_n);
    hpxcl_single_mul_kernel.set_arg(2, hpxcl_single_buffer_o);
    
    // set kernel args for log
    hpxcl_single_log_kernel.set_arg(0, hpxcl_single_buffer_z);
    hpxcl_single_log_kernel.set_arg(1, hpxcl_single_buffer_p);
    
}

static std::vector<char> hpxcl_single_calculate(std::vector<float> &a,
                                                std::vector<float> &b,
                                                std::vector<float> &c,
                                                double* t_nonblock,
                                                double* t_sync,
                                                double* t_finish)
{
    // start time measurement
    timer_start();
    
    // do nothing if matrices are wrong
    if(a.size() != b.size() || b.size() != c.size())
    {
        return std::vector<char>();
    }

    size_t size = a.size();

    // copy data to gpu
    shared_future<event> write_a_event = 
               hpxcl_single_buffer_a.enqueue_write(0, size*sizeof(float), &a[0]);
    shared_future<event> write_b_event =
               hpxcl_single_buffer_b.enqueue_write(0, size*sizeof(float), &b[0]);
    shared_future<event> write_c_event =
               hpxcl_single_buffer_c.enqueue_write(0, size*sizeof(float), &c[0]);

    // set work dimensions
    work_size<1> dim;
    dim[0].offset = 0;
    dim[0].size = size;

    // run exp kernel
    shared_future<event> kernel_exp_event =
                             hpxcl_single_exp_kernel.enqueue(dim, write_b_event);

    // run add kernel
    std::vector<shared_future<event>> add_dependencies;
    add_dependencies.push_back(kernel_exp_event);
    add_dependencies.push_back(write_a_event);
    shared_future<event> kernel_add_event = 
                          hpxcl_single_add_kernel.enqueue(dim, add_dependencies);

    // run dbl kernel
    shared_future<event> kernel_dbl_event =
                             hpxcl_single_dbl_kernel.enqueue(dim, write_c_event);

    // run mul kernel
    std::vector<shared_future<event>> mul_dependencies;
    mul_dependencies.push_back(kernel_add_event);
    mul_dependencies.push_back(kernel_dbl_event);
    shared_future<event> kernel_mul_event = 
                          hpxcl_single_mul_kernel.enqueue(dim, mul_dependencies);

    // run log kernel
    shared_future<event> kernel_log_event = 
                          hpxcl_single_log_kernel.enqueue(dim, kernel_mul_event);

    // enqueue result read
    shared_future<event> read_event_future = 
                        hpxcl_single_buffer_z.enqueue_read(0, size*sizeof(float),
                                                          kernel_log_event);

    
    ////////// UNTIL HERE ALL CALLS WERE NON-BLOCKING /////////////////////////

    // get time of non-blocking calls
    *t_nonblock = timer_stop();

    // wait for enqueue_read to return the event
    event read_event = read_event_future.get();

    // get time of synchronization
    *t_sync = timer_stop();

    // wait for calculation to complete and return data
    boost::shared_ptr<std::vector<char>> data_ptr = read_event.get_data().get();

    // get total time of execution
    *t_finish = timer_stop();
    
    // return the computed data
    return *data_ptr;


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





















#endif //BENCHMARK_HPXCL_SINGLE_HPP__

