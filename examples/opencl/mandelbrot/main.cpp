// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/apply.hpp>

#include "../../../opencl.hpp"

#include "mandelbrotworker.hpp"
#include "workload.hpp"
#include "work_queue.hpp"
#include "pngwriter.hpp"

#include <string>
#include <boost/shared_ptr.hpp>

int hpx_main(int argc, char* argv[])
{

    // Print help message on wrong argument count
    if(argc < 1)
    {
        hpx::cerr << "Usage: " << argv[0] << hpx::endl;//" matrixsize" << hpx::endl;
        return hpx::finalize();
    }


    // The main scope
    {

        // get local devices
        std::vector<hpx::opencl::device> devices =
                                hpx::opencl::get_devices(hpx::find_here(),
                                                         CL_DEVICE_TYPE_ALL,
                                                         1.1f).get();

        // Check whether there are any devices
        if(devices.size() < 2)
        {
            hpx::cerr << "No OpenCL devices found!" << hpx::endl;
            return hpx::finalize();
        }

        // create workqueue
        boost::shared_ptr<work_queue<boost::shared_ptr<workload>>>
                       workqueue(new work_queue<boost::shared_ptr<workload>>()); 

        // create one worker
        hpx::cout << "starting worker 1 ..." << hpx::endl;
        mandelbrotworker worker1(devices[0], workqueue);
/*        hpx::cout << "starting worker 2 ..." << hpx::endl;
        mandelbrotworker worker2(devices[0], workqueue);
        hpx::cout << "starting worker 3 ..." << hpx::endl;
        mandelbrotworker worker3(devices[0], workqueue);
        hpx::cout << "starting worker 4 ..." << hpx::endl;
        mandelbrotworker worker4(devices[0], workqueue);
        hpx::cout << "starting worker 5 ..." << hpx::endl;
        mandelbrotworker worker5(devices[0], workqueue);
        hpx::cout << "starting worker 6 ..." << hpx::endl;
        mandelbrotworker worker6(devices[0], workqueue);
        hpx::cout << "starting worker 7 ..." << hpx::endl;
        mandelbrotworker worker7(devices[0], workqueue);
        hpx::cout << "starting worker 8 ..." << hpx::endl;
        mandelbrotworker worker8(devices[0], workqueue);
        hpx::cout << "starting worker 9 ..." << hpx::endl;
        mandelbrotworker worker9(devices[0], workqueue);
        hpx::cout << "starting worker 10 ..." << hpx::endl;
        mandelbrotworker worker10(devices[0], workqueue);
*/


        
        double left = -0.743643887070496004085;
        double right = -0.743643887003805995915;
        double top = 0.131825904230338753064;
        double bottom = 0.131825904180321246936;
        /*
        double left = -2.238461538;
        double right = 0.8384615385;
        double top = 1.153846154;
        double bottom = -1.153846154;
        */
        size_t img_x = 960;
        size_t img_y = 720;

        // create an image
        hpx::cout << "creating img ..." << hpx::endl;
        unsigned long img = png_create(img_x,img_y);


        // add workloads for all lines
        for(size_t i = 0; i < img_y; i++)
        {
            hpx::cout << "adding line " << i << " ..." << hpx::endl;
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
        while(workqueue->retrieve_finished_work(&done_row))
        {

            hpx::cout << "taking line " << done_row->pos_in_img << " ... " << hpx::endl;
            png_set_row(img, done_row->pos_in_img, done_row->pixeldata.data());


        }

        // wait for worker to finish
        hpx::cout << "waiting for worker 1 to finish ..." << hpx::endl;
        worker1.join();
/*        hpx::cout << "waiting for worker 2 to finish ..." << hpx::endl;
        worker2.join();
        hpx::cout << "waiting for worker 3 to finish ..." << hpx::endl;
        worker3.join();
        hpx::cout << "waiting for worker 4 to finish ..." << hpx::endl;
        worker4.join();
        hpx::cout << "waiting for worker 5 to finish ..." << hpx::endl;
        worker5.join();
        hpx::cout << "waiting for worker 6 to finish ..." << hpx::endl;
        worker6.join();
        hpx::cout << "waiting for worker 7 to finish ..." << hpx::endl;
        worker7.join();
        hpx::cout << "waiting for worker 8 to finish ..." << hpx::endl;
        worker8.join();
        hpx::cout << "waiting for worker 9 to finish ..." << hpx::endl;
        worker9.join();
        hpx::cout << "waiting for worker 10 to finish ..." << hpx::endl;
        worker10.join();
*/
        // save the png
        hpx::cout << "saving png ..." << hpx::endl;
        png_save_and_close(img, "test.png");

    }

    hpx::cout << "Program finished." << hpx::endl;
   
    // End the program
    return hpx::finalize();

}

