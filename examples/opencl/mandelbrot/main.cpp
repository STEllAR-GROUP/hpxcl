// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/apply.hpp>
#include <hpx/util/static.hpp>

#include "../../../opencl.hpp"

#include "pngwriter.hpp"
#include "timer.hpp"
#include "image_generator.hpp"

#include <string>
#include <boost/shared_ptr.hpp>

int hpx_main(boost::program_options::variables_map & vm)
{

    std::size_t num_kernels = 0;
    bool verbose = false;

    // Print help message on wrong argument count
    if (vm.count("num-parallel-kernels"))
        num_kernels = vm["num-parallel-kernels"].as<std::size_t>();
    if (vm.count("v"))
        verbose = true;

    // The main scope
    {

        // get all HPX localities
        if(verbose) hpx::cout << "Finding all hpx localities ..." << hpx::endl;
        std::vector<hpx::naming::id_type> localities = 
                                            hpx::find_all_localities();
        if(verbose) hpx::cout << localities.size() << " hpx localities found!" << hpx::endl; 

        // query all devices
        if(verbose) hpx::cout << "Requesting device lists from localities ..." << hpx::endl;
        std::vector<hpx::lcos::shared_future<std::vector<hpx::opencl::device>>>
        locality_device_futures;
        BOOST_FOREACH(hpx::naming::id_type & locality, localities)
        {

            // get all devices on locality
            hpx::lcos::shared_future<std::vector<hpx::opencl::device>>
            locality_device_future = hpx::opencl::get_devices(locality,
                                                             CL_DEVICE_TYPE_GPU,
                                                             1.1f);

            // add locality device future to list of futures
            locality_device_futures.push_back(locality_device_future);

        }
        
        // wait for all localities to respond, then add all devices to devicelist
        if(verbose) hpx::cout << "Waiting for device lists ..." << hpx::endl;
        boost::shared_ptr<std::vector<hpx::opencl::device>>
                devices(new std::vector<hpx::opencl::device>());
        BOOST_FOREACH(
                    hpx::lcos::shared_future<std::vector<hpx::opencl::device>>
                    locality_device_future,
                    locality_device_futures)
        {

            // wait for device query to finish
            std::vector<hpx::opencl::device> locality_devices = 
                                                   locality_device_future.get();

            // add all devices to device list
            devices->insert(devices->end(), locality_devices.begin(),
                                          locality_devices.end());

        }

        // Check whether there are any devices
        if(devices->size() < 1)
        {
            hpx::cerr << "No OpenCL devices found!" << hpx::endl;
            return hpx::finalize();
        }
        else
        {
            hpx::cout << devices->size() << " OpenCL devices found!" << hpx::endl;
        }

        /*
        double left = -0.743643887062801003142;
        double right = -0.743643887011500996858;
        double top = 0.131825904224567502357;
        double bottom = 0.131825904186092497643;
        */
        


        /*
        double left = -2.238461538;
        double right = 0.8384615385;
        double top = 1.153846154;
        double bottom = -1.153846154;
        */
        size_t img_x = 2560;
        size_t img_y = 1920;

        // create image_generator
        image_generator img_gen(devices, img_x, num_kernels, verbose);
        
        // wait for workers to finish initialization
        if(verbose) hpx::cout << "waiting for workers to finish startup ..." << hpx::endl;
        img_gen.wait_for_startup_finished();

        if(verbose) hpx::cout << "adding queuing image ..." << hpx::endl;
        // start timer
        timer_start();

        // queue image
        img_gen.compute_image(0.0,0.0,0.0,0.0,img_x, img_y, true).get();
        
        // stop timer
        double time = timer_stop();

        hpx::cout << "time: " << time << " ms" << hpx::endl;

        // end the image generator
        img_gen.shutdown();

        // save the png

    }

    if(verbose) hpx::cout << "Program finished." << hpx::endl;
   
    // End the program
    return hpx::finalize();

}



//////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    boost::program_options::options_description cmdline(
                                "Usage: " HPX_APPLICATION_STRING " [options]");
    cmdline.add_options()
        ( "num-parallel-kernels"
        , boost::program_options::value<std::size_t>()->default_value(10)
        , "the number of parallel kernel invocations per gpu") ;

    cmdline.add_options()
        ( "v"
        , "verbose output") ;

    return hpx::init(cmdline, argc, argv);
}
