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

#include "mandelbrotworker.hpp"
#include "workload.hpp"
#include "work_queue.hpp"
#include "pngwriter.hpp"
#include "timer.hpp"

#include <string>
#include <boost/shared_ptr.hpp>

int hpx_main(boost::program_options::variables_map & vm)
{

    std::size_t num_kernels = 0;

    // Print help message on wrong argument count
    if (vm.count("num_parallel_kernels"))
        num_kernels = vm["num_parallel_kernels"].as<std::size_t>();

    // The main scope
    {

        // get all HPX localities
        hpx::cout << "Finding all hpx localities ..." << hpx::endl;
        std::vector<hpx::naming::id_type> localities = 
                                            hpx::find_all_localities();
        hpx::cout << localities.size() << " hpx localities found!" << hpx::endl; 

        // query all devices
        hpx::cout << "Requesting device lists from localities ..." << hpx::endl;
        std::vector<hpx::lcos::shared_future<std::vector<hpx::opencl::device>>>
        locality_device_futures;
        BOOST_FOREACH(hpx::naming::id_type & locality, localities)
        {

            // get all devices on locality
            hpx::lcos::shared_future<std::vector<hpx::opencl::device>>
            locality_device_future = hpx::opencl::get_devices(locality,
                                                             CL_DEVICE_TYPE_GPU,
                                                             1.0f);

            // add locality device future to list of futures
            locality_device_futures.push_back(locality_device_future);

        }
        
        // wait for all localities to respond, then add all devices to devicelist
        hpx::cout << "Waiting for device lists ..." << hpx::endl;
        std::vector<hpx::opencl::device> devices;
        BOOST_FOREACH(
                    hpx::lcos::shared_future<std::vector<hpx::opencl::device>>
                    locality_device_future,
                    locality_device_futures)
        {

            // wait for device query to finish
            std::vector<hpx::opencl::device> locality_devices = 
                                                   locality_device_future.get();

            // add all devices to device list
            devices.insert(devices.end(), locality_devices.begin(),
                                          locality_devices.end());

        }

        // Check whether there are any devices
        if(devices.size() < 1)
        {
            hpx::cerr << "No OpenCL devices found!" << hpx::endl;
            return hpx::finalize();
        }
        else
        {
            hpx::cout << devices.size() << " OpenCL devices found!" << hpx::endl;
        }

        // create workqueue
        boost::shared_ptr<work_queue<boost::shared_ptr<workload>>>
                       workqueue(new work_queue<boost::shared_ptr<workload>>()); 

        // create workers
        hpx::cout << "starting workers ..." << hpx::endl;
        std::vector<boost::shared_ptr<mandelbrotworker>> workers;
        BOOST_FOREACH(hpx::opencl::device & device, devices)
        {
            
            // create worker
            boost::shared_ptr<mandelbrotworker> worker(
                                new mandelbrotworker(device,
                                                     workqueue,
                                                     num_kernels));
            // add worker to workerlist
            workers.push_back(worker);

        }
        
        /*
        double left = -0.743643887062801003142;
        double right = -0.743643887011500996858;
        double top = 0.131825904224567502357;
        double bottom = 0.131825904186092497643;
        */
        


        ///*
        double left = -2.238461538;
        double right = 0.8384615385;
        double top = 1.153846154;
        double bottom = -1.153846154;
        //*/
        size_t img_x = 2560;
        size_t img_y = 1920;

        // create an image
        hpx::cout << "creating img ..." << hpx::endl;
        unsigned long img = png_create(img_x,img_y);

        // wait for workers to finish initialization
        hpx::cout << "waiting for workers to finish startup ..." << hpx::endl;
        BOOST_FOREACH(boost::shared_ptr<mandelbrotworker> worker, workers)
        {

            worker->wait_for_startup_finished();

        }

        hpx::cout << "adding workpackets to workqueue ..." << hpx::endl;
        timer_start();
        // add workloads for all lines
        for(size_t i = 0; i < img_y; i++)
        {
            // hpx::cout << "adding line " << i << " ..." << hpx::endl;
            boost::shared_ptr<workload> row(
                   new workload(img_x, left, top - (top - bottom)*i/(img_y - 1), right-left, 0.0, 0, i));
            workqueue->add_work(row);

        }
        
        // finish workqueue
        hpx::cout << "finishing workqueue ..." << hpx::endl;
        workqueue->finish();

        // enter calculated rows to image
        hpx::cout << "waiting for lines ..." << hpx::endl;
        boost::shared_ptr<workload> done_row;
        int i = 0;
        while(workqueue->retrieve_finished_work(&done_row))
        {

            // hpx::cout << "taking line " << done_row->pos_in_img << " ... " << hpx::endl;
            png_set_row(img, done_row->pos_in_img, done_row->pixeldata.data());
            hpx::cout << "progress: " << (++i) << " / " << img_y << hpx::endl;

        }

        // wait for worker to finish
        hpx::cout << "waiting for workers to finish ..." << hpx::endl;
        BOOST_FOREACH(boost::shared_ptr<mandelbrotworker> worker, workers)
        {

            worker->join();

        }
       
        double time = timer_stop();

        hpx::cout << "time: " << time << " ms" << hpx::endl;

        // save the png
        hpx::cout << "saving png ..." << hpx::endl;
        png_save_and_close(img, "test.png");

    }

    hpx::cout << "Program finished." << hpx::endl;
   
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
        ( "num_parallel_kernels"
        , boost::program_options::value<std::size_t>()->default_value(10)
        , "the number of parallel kernel invocations") ;

    return hpx::init(cmdline, argc, argv);
}
