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



void grainsize_bench(std::vector<hpx::opencl::device> devices,
                   size_t img_x, size_t img_y,
                   size_t num_iterations, size_t num_kernels, bool verbose)
{
    
            // default benchmark image
            double posx = -0.743643887037151;
            double posy = 0.131825904205330;
            double zoom = 6.2426215349789484160e10;

            hpx::cerr << "Starting in benchmark mode." << hpx::endl;

            // create image generator without gpus
            image_generator img_gen(1, 1, num_kernels, verbose, devices);

            img_gen.wait_for_startup_finished();

            // iterate through all configurations
            for(size_t grainsize = 256; grainsize <= img_x*img_y; grainsize *= 2)
            {
                hpx::cerr << "Starting test with grainsize " << grainsize << " ..."
                          << hpx::endl;
               
                // calculate tile size
                size_t tilesize_x, tilesize_y;
                if(grainsize < img_x)
                {
                    tilesize_y = 1;
                    tilesize_x = grainsize;
                }
                else
                {
                    tilesize_x = img_x;
                    tilesize_y = grainsize/img_x;
                }
               
                hpx::cerr << "Using tilesizes: " << tilesize_x << "x"
                          << tilesize_y << hpx::endl;

                // initialize timer
                double total_time = 0.0;

                // main benchmark loop
                for(size_t i = 0; i < num_iterations + 1; i++)
                {
                    if(i == 0)
                    {
                        hpx::cerr << "Warmup iteration ..." << hpx::endl;
                    }
                    if(i >= 1)
                    {
                        hpx::cerr << "Starting benchmark iteration " << i << "/"
                                  << num_iterations << " ..." << hpx::endl;

                        // start time measurement
                        timer_start();
                    }

                    img_gen.compute_image(posx,
                                          posy,
                                          zoom,
                                          0.0,
                                          img_x,
                                          img_y,
                                          true,
                                          tilesize_x,
                                          tilesize_y).get();

                    if(i >= 1)
                    {

                        // measure time
                        total_time += timer_stop();

                    }
                }

                // calculate average time
                double time = total_time / (double)num_iterations;
                
                hpx::cerr << "Time: " << time << " ms" << hpx::endl;
                hpx::cout << grainsize
                          << "\t" << time 
                          << hpx::endl;

            }

            hpx::cerr << "Done." << hpx::endl;
            img_gen.shutdown();

   
    
    
}

void speedup_bench(std::vector<hpx::opencl::device> devices,
                   size_t tilesize_x, size_t tilesize_y, size_t img_x, size_t img_y,
                   size_t num_iterations, size_t num_kernels, bool verbose)
{
    
            // default benchmark image
            double posx = -0.743643887037151;
            double posy = 0.131825904205330;
            double zoom = 6.2426215349789484160e10;

            hpx::cerr << "Starting in benchmark mode." << hpx::endl;

            // create image generator without gpus
            image_generator img_gen(tilesize_x, tilesize_y, num_kernels, verbose);

            // save the time for single-gpu
            double single_gpu_time = 1.0f;

            for(size_t num_gpus = 1; num_gpus <= devices.size(); num_gpus++)
            {
                hpx::cerr << "Starting test with " << num_gpus << " gpus ..."
                          << hpx::endl;
               
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

                // main benchmark loop
                for(size_t i = 0; i < num_iterations + 1; i++)
                {
                    // start timer after first iteration (warmup iteration)
                    if(i == 1) timer_start();
                    img_gen.compute_image(posx,
                                          posy,
                                          zoom,
                                          0.0,
                                          img_x,
                                          img_y,
                                          true,
                                          tilesize_x,
                                          tilesize_y).get();
                }

                // stop timer
                double time = timer_stop();
                time = time / (double)num_iterations - 1;
                
                // save time if we only have one gpu
                if(num_gpus == 1)
                    single_gpu_time = time;
                
                hpx::cerr << "Time: " << time << " ms" << hpx::endl;
                hpx::cout << num_gpus 
                          << "\t" << time 
                          << "\t" << (single_gpu_time / time) 
                          << "\t" << ((single_gpu_time / time) / (double)num_gpus)
                          << hpx::endl;

            }

            hpx::cerr << "Done." << hpx::endl;
            img_gen.shutdown();

   
    
    
}


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
            
            /*speedup_bench(devices,
                          3840, 8,
                          3840, 2160,
                          10,
                          4,
                          verbose);
*/
            grainsize_bench(devices,
                            2048, 1024,
                            10,
                            4,
                            verbose);
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
        , "runs benchmark") ;

    return hpx::init(cmdline, argc, argv);
}
