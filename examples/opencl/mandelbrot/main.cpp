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
    bool benchmark = false;

    // Print help message on wrong argument count
    if (vm.count("num-parallel-kernels"))
        num_kernels = vm["num-parallel-kernels"].as<std::size_t>();
    if (vm.count("v"))
        verbose = true;
    if (vm.count("bench"))
        benchmark = true;

    // The main scope
    {

        // get all devices
        std::vector<hpx::opencl::device> devices = 
           hpx::opencl::get_all_devices(CL_DEVICE_TYPE_GPU, "OpenCL 1.1").get();

        // Check whether there are any devices
        if(devices.size() < 1)
        {
            hpx::cerr << "No OpenCL devices found!" << hpx::endl;
            return hpx::finalize();
        }
        else
        {
            hpx::cerr << devices.size() << " OpenCL devices found!" << hpx::endl;
        }


        //double posx = -0.7;
        //double posy = 0.0;
        //double zoom = 1.04;
        ////double zoom = 0.05658352842407526628;

        double posx = -0.743643887037151;
        double posy = 0.131825904205330;
        double zoom = 6.2426215349789484160e10;
        //double zoom = 35.8603219463046942295;

        size_t img_x = 3840;
        size_t img_y = 2160;

        if(!benchmark)
        {
            // create image_generator
            image_generator img_gen(img_x, 8, num_kernels, verbose, devices);
            
            // wait for workers to finish initialization
            if(verbose) hpx::cout << "waiting for workers to finish startup ..." << hpx::endl;
            img_gen.wait_for_startup_finished();
    
            // start timer
            timer_start();
    
            // queue image
            boost::shared_ptr<std::vector<char>> img_data =
                img_gen.compute_image(posx,
                                      posy,
                                      zoom,
                                      0.0,
                                      img_x,
                                      img_y,
                                      false,
                                      img_x,
                                      4).get();
            
            // stop timer
            double time = timer_stop();
    
            hpx::cout << "time: " << time << " ms" << hpx::endl;
    
            // end the image generator
            img_gen.shutdown();
    
            // save the png
            save_png(img_data, img_x, img_y, "test.png");
        
        } else {

            size_t num_iterations = 10;

            std::cerr << "Starting in benchmark mode." << std::endl;
            size_t chunksize = 8;

            // create image generator without gpus
            image_generator img_gen(img_x, chunksize, num_kernels, verbose);

            // save the time for single-gpu
            double single_gpu_time = 1.0f;

            for(size_t num_gpus = 1; num_gpus <= devices.size(); num_gpus++)
            {
                std::cerr << "Starting test with " << num_gpus << " gpus ..."
                          << std::endl;
               
                // Add another worker
                if(verbose){
                     hpx::cerr << "adding worker ..."
                               << hpx::endl;
                }
                img_gen.add_worker(devices[devices.size() - num_gpus], 4); 

                // Wait for the worker to initialize
                if(verbose){
                     hpx::cerr << "waiting for worker to finish startup ..."
                               << hpx::endl;
                }
                img_gen.wait_for_startup_finished();

                // Start timing

                // Add image
                for(size_t i = 0; i < num_iterations + 1; i++)
                {
                    if(i == 1) timer_start();
                    img_gen.compute_image(posx,
                                          posy,
                                          zoom,
                                          0.0,
                                          img_x,
                                          img_y,
                                          true,
                                          img_x,
                                          chunksize).get();
                }

                // stop timer
                double time = timer_stop();
                time = time / (double)num_iterations - 1;

                // save time if we only have one gpu
                if(num_gpus == 1)
                    single_gpu_time = time;

                
                std::cerr << "Time: " << time << " ms" << hpx::endl;
                std::cout << num_gpus 
                          << "\t" << time 
                          << "\t" << (single_gpu_time / time) 
                          << "\t" << ((single_gpu_time / time) / (double)num_gpus)
                          << hpx::endl;

            }

            std::cerr << "Done." << hpx::endl;
            img_gen.shutdown();

        }

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
        , boost::program_options::value<std::size_t>()->default_value(4)
        , "the number of parallel kernel invocations per gpu") ;

    cmdline.add_options()
        ( "v"
        , "verbose output") ;

    cmdline.add_options()
        ( "bench"
        , "disables writing to files") ;

    return hpx::init(cmdline, argc, argv);
}
