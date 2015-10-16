// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/apply.hpp>
#include <hpx/util/static.hpp>

#include "../../../../opencl.hpp"

#include "maps_image_generator.hpp"

#include "requesthandler.hpp"
#include "webserver.hpp"
//#include "../maps_webserver.hpp"


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
            hpx::opencl::create_all_devices(
                        CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR,
                        "OpenCL 1.1").get();

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


        size_t tilesize_x = 256;
        size_t tilesize_y = 256;
        size_t lines_per_gpu = 32;

        // generate requesthandler, will order requests and convert coordinates
        hpx::opencl::examples::mandelbrot::requesthandler requesthandler(
                                                            tilesize_x,
                                                            tilesize_y,
                                                            lines_per_gpu);

   
        // create image_generator
        hpx::opencl::examples::mandelbrot::maps_image_generator
           img_gen(tilesize_x,
                   lines_per_gpu,
                   num_kernels,
                   verbose,
                   boost::bind(
                     &hpx::opencl::examples::mandelbrot::requesthandler::query_request,
                     &requesthandler),
                   devices);
        
        // wait for workers to finish initialization
        if(verbose) hpx::cout << "waiting for workers to finish startup ..." << hpx::endl;
        img_gen.wait_for_startup_finished();

        hpx::cout << "Starting webservers ..." << hpx::endl;
   
        // generate webserver
        hpx::opencl::examples::mandelbrot::webserver webserver(8080,
                                                               &requesthandler);

        // start the webserver
        webserver.start();

        while(true)
        {
            hpx::this_thread::sleep_for(boost::posix_time::milliseconds(1000));
        }
        
        webserver.stop();
        
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
        , boost::program_options::value<std::size_t>()->default_value(3)
        , "the number of parallel kernel invocations per gpu") ;

    cmdline.add_options()
        ( "v"
        , "verbose output") ;

    return hpx::init(cmdline, argc, argv);
}
