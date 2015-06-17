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
hpx::opencl::tests::performance::testresults results;
static buffer_type test_data;
static std::size_t num_iterations;


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

static void run_hpx_loopback_test( hpx::naming::id_type target_location )
{

    results.start_test("HPX_remote", "GB/s");
   
    const std::size_t data_transfer_per_test =
        test_data.size() * 2 * num_iterations;


    double throughput_gbps = 0.0;
    do
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

        const double duration = walltime.elapsed();
       
        ensure_valid(result_data); 

        const double throughput = data_transfer_per_test / duration;

        throughput_gbps = throughput/(1024.0*1024.0*1024.0);
        
    } while(results.add(throughput_gbps));

}



static void cl_test(hpx::opencl::device local_device,
                    hpx::opencl::device remote_device)
{


    const std::size_t testdata_size = static_cast<std::size_t>(1) << 22;
    num_iterations = 50;

    // Get localities
    hpx::naming::id_type remote_location =
        hpx::get_colocation_id_sync(remote_device.get_gid());
    hpx::naming::id_type local_location =
        hpx::get_colocation_id_sync(local_device.get_gid());
    if(local_location != hpx::find_here())
        die("Internal ERROR! local_location is not here.");

    // Generate random vector
    std::cout << "Generating test data ..." << std::endl;
    test_data = buffer_type ( new char[testdata_size], testdata_size,
                              buffer_type::init_mode::take );
    std::cout << "Test data generated." << std::endl;
    for(std::size_t i = 0; i < testdata_size; i++){
        test_data[i] = static_cast<char>(rand());
    }

    // Run hpx loopback test
    run_hpx_loopback_test(remote_location);




    results.start_test("test", "GFLOPS");
    results.add(10);
    results.add(11);
    results.add(12);
    results.add(13);
    results.add(14);
    results.add(15);
    results.add(16);
    results.add(17);
    results.add(18);
    results.add(19);


    std::cout << results << std::endl;
}



