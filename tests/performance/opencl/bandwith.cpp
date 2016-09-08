// Copyright (c)       2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "util/cl_tests.hpp"

#include "util/testresults.hpp"

#include <hpx/util/high_resolution_timer.hpp>

#include <cstdlib>

typedef hpx::serialization::serialize_buffer<char> buffer_type;


// global variables
static buffer_type test_data;

buffer_type
loopback(buffer_type buf){
    return buf;
}

HPX_PLAIN_ACTION(loopback, loopback_action);

static void ensure_valid( buffer_type result )
{
    if( result.size() != test_data.size() ){
        die("result size is wrong!");
    }

    for( std::size_t i = 0; i < result.size(); i++ ){
        if(test_data[i] != result[i])
            die("result is wrong!");
    }
}


static void run_opencl_local_test( hpx::opencl::device device )
{

    hpx::opencl::buffer buffer =
        device.create_buffer(CL_MEM_READ_WRITE, test_data.size());

    auto device_ptr =
        hpx::get_ptr<hpx::opencl::server::device>(device.get_id()).get();
    auto buffer_ptr =
        hpx::get_ptr<hpx::opencl::server::buffer>(buffer.get_id()).get();


    cl_context context = device_ptr->get_context();
    cl_command_queue write_command_queue = device_ptr->get_write_command_queue();
    cl_command_queue read_command_queue = device_ptr->get_read_command_queue();
    cl_mem buffer_id = buffer_ptr->get_cl_mem();

    std::map<std::string, std::string> atts;
    atts["size"] = std::to_string(test_data.size());
    atts["iterations"] = std::to_string(num_iterations);
    results.start_test("OpenCL_local_host_to_local_device", "GB/s", atts);

    const std::size_t data_transfer_per_test =
        test_data.size() * 2 * num_iterations;


    double throughput_gbps = 0.0;
    while(results.needs_more_testing())
    {
        // initialize the buffer
        buffer_type buf ( test_data.size() );
        std::copy(test_data.data(), test_data.data()+test_data.size(), buf.data());

        cl_int err;

        err = clFinish(read_command_queue);
        cl_ensure(err, "clFinish()");
        err = clFinish(write_command_queue);
        cl_ensure(err, "clFinish()");

        hpx::util::high_resolution_timer walltime;
        for(std::size_t it = 0; it < num_iterations; it ++)
        {
            err = clEnqueueWriteBuffer( write_command_queue,
                                        buffer_id,
                                        CL_TRUE,
                                        0,
                                        buf.size(),
                                        buf.data(),
                                        0, NULL, NULL );
            cl_ensure(err, "clEnqueueWriteBuffer()");

            err = clEnqueueReadBuffer( read_command_queue,
                                       buffer_id,
                                       CL_TRUE,
                                       0,
                                       buf.size(),
                                       buf.data(),
                                       0, NULL, NULL );
            cl_ensure(err, "clEnqueueReadBuffer()");
        }


        const double duration = walltime.elapsed();
        ensure_valid(buf);

        const double throughput = data_transfer_per_test / duration;
        throughput_gbps = throughput/(1024.0*1024.0*1024.0);

        results.add(throughput_gbps);
    }



}


static void run_opencl_local_send_test( hpx::opencl::device device )
{

    hpx::opencl::buffer buffer1 =
        device.create_buffer(CL_MEM_READ_WRITE, test_data.size());
    hpx::opencl::buffer buffer2 =
        device.create_buffer(CL_MEM_READ_WRITE, test_data.size());

    auto device_ptr =
        hpx::get_ptr<hpx::opencl::server::device>(device.get_id()).get();
    auto buffer1_ptr =
        hpx::get_ptr<hpx::opencl::server::buffer>(buffer1.get_id()).get();
    auto buffer2_ptr =
        hpx::get_ptr<hpx::opencl::server::buffer>(buffer2.get_id()).get();


    cl_context context = device_ptr->get_context();
    cl_command_queue command_queue = device_ptr->get_write_command_queue();
    cl_mem buffer1_id = buffer1_ptr->get_cl_mem();
    cl_mem buffer2_id = buffer2_ptr->get_cl_mem();

    std::map<std::string, std::string> atts;
    atts["size"] = std::to_string(test_data.size());
    atts["iterations"] = std::to_string(num_iterations);
    results.start_test("OpenCL_local_device_to_local_device", "GB/s", atts);

    const std::size_t data_transfer_per_test =
        test_data.size() * 2 * num_iterations;


    double throughput_gbps = 0.0;
    while(results.needs_more_testing())
    {
        // initialize the buffer
        buffer_type buf ( test_data.size() );
        std::copy(test_data.data(), test_data.data()+test_data.size(), buf.data());

        cl_int err;
        cl_event event, new_event;

        err = clEnqueueWriteBuffer( command_queue,
                                    buffer1_id,
                                    CL_TRUE,
                                    0,
                                    buf.size(),
                                    buf.data(),
                                    0, NULL, &event );
        cl_ensure(err, "clEnqueueWriteBuffer()");

        err = clFinish(command_queue);
        cl_ensure(err, "clFinish()");

        hpx::util::high_resolution_timer walltime;
        for(std::size_t it = 0; it < num_iterations; it ++)
        {
            err = clEnqueueCopyBuffer( command_queue,
                                       buffer1_id,
                                       buffer2_id,
                                       0, 0,
                                       buf.size(),
                                       1, &event,
                                       &new_event );
            cl_ensure(err, "clEnqueueCopyBuffer()");

            err = clReleaseEvent(event);
            cl_ensure(err, "clReleaseEvent()");
            event = new_event;

            err = clEnqueueCopyBuffer( command_queue,
                                       buffer2_id,
                                       buffer1_id,
                                       0, 0,
                                       buf.size(),
                                       1, &event,
                                       &new_event );
            cl_ensure(err, "clEnqueueCopyBuffer()");

            err = clReleaseEvent(event);
            cl_ensure(err, "clReleaseEvent()");
            event = new_event;
        }

        err = clWaitForEvents(1, &event);
        cl_ensure(err, "clWaitForEvents()");

        const double duration = walltime.elapsed();

        err = clReleaseEvent(event);
        cl_ensure(err, "clReleaseEvent()");

        err = clEnqueueReadBuffer( command_queue,
                                   buffer1_id,
                                   CL_TRUE,
                                   0,
                                   buf.size(),
                                   buf.data(),
                                   0, NULL, NULL );
        cl_ensure(err, "clEnqueueReadBuffer()");
        ensure_valid(buf);

        err = clEnqueueReadBuffer( command_queue,
                                   buffer2_id,
                                   CL_TRUE,
                                   0,
                                   buf.size(),
                                   buf.data(),
                                   0, NULL, NULL );
        cl_ensure(err, "clEnqueueReadBuffer()");
        ensure_valid(buf);

        const double throughput = data_transfer_per_test / duration;
        throughput_gbps = throughput/(1024.0*1024.0*1024.0);

        results.add(throughput_gbps);
    }



}


static void run_hpxcl_send_test( hpx::opencl::device device1,
                                 hpx::opencl::device device2 )
{

    hpx::opencl::buffer buffer1 =
        device1.create_buffer(CL_MEM_READ_WRITE, test_data.size());
    hpx::opencl::buffer buffer2 =
        device2.create_buffer(CL_MEM_READ_WRITE, test_data.size());



    std::string device1_location = "remote";
    if(hpx::get_colocation_id(hpx::launch::sync, device1.get_id()) == hpx::find_here())
        device1_location = "local";
    std::string device2_location = "remote";
    if(hpx::get_colocation_id(hpx::launch::sync, device2.get_id()) == hpx::find_here())
        device2_location = "local";

    std::map<std::string, std::string> atts;
    atts["size"] = std::to_string(test_data.size());
    atts["iterations"] = std::to_string(num_iterations);
    results.start_test("HPXCL_" + device1_location + "_device_to_" +
                        device2_location + "_device",
                       "GB/s", atts);

    const std::size_t data_transfer_per_test =
        test_data.size() * 2 * num_iterations;

    double throughput_gbps = 0.0;
    while(results.needs_more_testing())
    {
        // initialize the buffer
        hpx::future<void> fut = buffer1.enqueue_write(0, test_data);

        fut.wait();

        // RUN!
        hpx::util::high_resolution_timer walltime;
        for(std::size_t it = 0; it < num_iterations; it ++)
        {
            // Copy from buffer1 to buffer2
            auto send_result =
                buffer1.enqueue_send(buffer2, 0, 0, test_data.size(), fut);

            // Copy from buffer2 to buffer1
            auto send_result2 =
                buffer2.enqueue_send(buffer1, 0, 0, test_data.size(),
                                     send_result.dst_future);
            fut = std::move(send_result2.dst_future);
        }

        // wait for last send to finish
        fut.get();

        // Measure elapsed time
        const double duration = walltime.elapsed();

        // Check if data is still valid
        ensure_valid(buffer1.enqueue_read(0, test_data.size()).get());

        // Calculate throughput
        const double throughput = data_transfer_per_test / duration;
        throughput_gbps = throughput/(1024.0*1024.0*1024.0);

        results.add(throughput_gbps);
    }

}

static void run_hpxcl_read_write_test( hpx::opencl::device device )
{

    hpx::opencl::buffer buffer =
        device.create_buffer(CL_MEM_READ_WRITE, test_data.size());



    std::map<std::string, std::string> atts;
    atts["size"] = std::to_string(test_data.size());
    atts["iterations"] = std::to_string(num_iterations);
    if(hpx::get_colocation_id(hpx::launch::sync, device.get_id()) == hpx::find_here())
        results.start_test("HPXCL_local_host_to_local_device", "GB/s", atts);
    else
        results.start_test("HPXCL_local_host_to_remote_device", "GB/s", atts);

    const std::size_t data_transfer_per_test =
        test_data.size() * 2 * num_iterations;

    double throughput_gbps = 0.0;
    while(results.needs_more_testing())
    {
        // initialize the buffer
        buffer_type read_buf ( test_data.size() );
        buffer_type write_buf ( test_data.size() );
        std::copy( test_data.data(), test_data.data()+test_data.size(),
                   write_buf.data() );

        // RUN!
        hpx::util::high_resolution_timer walltime;
        for(std::size_t it = 0; it < num_iterations; it ++)
        {
            // Copy to device
            auto fut_tmp = buffer.enqueue_write(0, write_buf);

            // Copy from device
            auto fut = buffer.enqueue_read(0, read_buf, fut_tmp);

            // Swap read and write buffer
            fut.get();
            std::swap(read_buf, write_buf);
        }

        // Measure elapsed time
        const double duration = walltime.elapsed();

        // Check if data is still valid
        ensure_valid(write_buf);

        // Calculate throughput
        const double throughput = data_transfer_per_test / duration;
        throughput_gbps = throughput/(1024.0*1024.0*1024.0);

        results.add(throughput_gbps);
    }

}

static void run_hpx_loopback_test( hpx::naming::id_type target_location )
{

    std::map<std::string, std::string> atts;
    atts["size"] = std::to_string(test_data.size());
    atts["iterations"] = std::to_string(num_iterations);
    results.start_test("HPX_local_host_to_remote_host", "GB/s", atts);

    const std::size_t data_transfer_per_test =
        test_data.size() * 2 * num_iterations;


    double throughput_gbps = 0.0;
    while(results.needs_more_testing())
    {
        hpx::util::high_resolution_timer walltime;

        buffer_type test_data_copy = test_data;
        hpx::future<buffer_type> tmp_result =
            hpx::make_ready_future<buffer_type>(std::move(test_data_copy));
        for(std::size_t it = 0; it < num_iterations; it ++)
        {
            tmp_result = tmp_result.then(
                [&target_location](hpx::future<buffer_type> && result){
                    return hpx::async<loopback_action>(target_location, result.get());
                });
        }
        buffer_type result_data = tmp_result.get();

        // measure time
        const double duration = walltime.elapsed();

        // make sure result is valid
        ensure_valid(result_data);

        // calculate the throughput
        const double throughput = data_transfer_per_test / duration;
        throughput_gbps = throughput/(1024.0*1024.0*1024.0);

        results.add(throughput_gbps);
    }

}



static void cl_test(hpx::opencl::device local_device,
                    hpx::opencl::device remote_device,
                    bool distributed)
{

    if(testdata_size == 0)
        testdata_size = static_cast<std::size_t>(1) << 20;
    if(num_iterations == 0)
        num_iterations = 50;

    // Get localities
    hpx::naming::id_type remote_location =
        hpx::get_colocation_id(hpx::launch::sync, remote_device.get_id());
    hpx::naming::id_type local_location =
        hpx::get_colocation_id(hpx::launch::sync, local_device.get_id());
    if(local_location != hpx::find_here())
        die("Internal ERROR! local_location is not here.");

    // Generate random vector
    std::cerr << "Generating test data ..." << std::endl;
    test_data = buffer_type ( testdata_size );
    std::cerr << "Test data generated." << std::endl;
    for(std::size_t i = 0; i < testdata_size; i++){
        test_data[i] = static_cast<char>(rand());
    }


    // Run local opencl test
    run_opencl_local_test(local_device);

    // Run local opencl send test
    run_opencl_local_send_test(local_device);

    if(distributed){
        // Run hpx loopback test
        run_hpx_loopback_test(remote_location);
    }

    // Run local hpxcl test
    run_hpxcl_read_write_test(local_device);

    if(distributed){
        // Run remote hpxcl test
        run_hpxcl_read_write_test(remote_device);
    }

    // Run hpxcl send local-local test
    run_hpxcl_send_test(local_device, local_device);

    if(distributed){
        // Run hpxcl send remote-remote test
        run_hpxcl_send_test(remote_device, remote_device);

        // Run hpxcl send local-remote test
        run_hpxcl_send_test(local_device, remote_device);
    }


}



