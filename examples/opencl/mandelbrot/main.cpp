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
            hpx::cout << devices.size() << " OpenCL devices found!" << hpx::endl;
        }


        double posx = -0.7;
        double posy = 0.0;
        double zoom = 1.04;
        ////double zoom = 0.05658352842407526628;

        //double posx = -0.743643887037151;
        //double posy = 0.131825904205330;
        //double zoom = 6.2426215349789484160e10;
        ////double zoom = 35.8603219463046942295;

        size_t img_x = 1920;
        size_t img_y = 1080;

        // create image_generator
        image_generator img_gen(devices, img_x, 4, num_kernels, verbose);
        
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

    return hpx::init(cmdline, argc, argv);
}
