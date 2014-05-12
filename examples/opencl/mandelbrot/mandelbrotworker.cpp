// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mandelbrotworker.hpp"
#include "mandelbrotkernel.hpp"

#include <cmath>

mandelbrotworker::mandelbrotworker(hpx::opencl::device device_,
                                   boost::shared_ptr<work_queue<
                                       boost::shared_ptr<workload>>> workqueue_)
                : workqueue(workqueue_), device(device_)
{

    // start worker
    worker_finished = hpx::async(&worker_main, workqueue, device); 

}


void
mandelbrotworker::join()
{

    // wait for worker to finish
    worker_finished.get();
    
}


void
mandelbrotworker::worker_main(
           boost::shared_ptr<work_queue<boost::shared_ptr<workload>>> workqueue,
           hpx::opencl::device device)
{
    try{
        // print device name
        hpx::cout << device.device_info_to_string(
                                  device.get_device_info(CL_DEVICE_VENDOR))
                  << ": "
                  << device.device_info_to_string(
                                  device.get_device_info(CL_DEVICE_NAME))
                  << hpx::endl;
    
        // build opencl program
        hpx::opencl::program mandelbrot_program =
                             device.create_program_with_source(mandelbrot_kernels);
        hpx::cout << "compiling" << hpx::endl;
        mandelbrot_program.build();
        hpx::cout << "creating kernel" << hpx::endl;
    
        // create kernel
        hpx::opencl::kernel kernel_noalias = 
                             mandelbrot_program.create_kernel("mandelbrot_noalias");
        hpx::cout << "kernel ready" << hpx::endl;
    
        // create output buffer
        hpx::opencl::buffer output_buffer = device.create_buffer(CL_MEM_WRITE_ONLY,
                                                               10000*sizeof(double));
        // create input buffer
        hpx::opencl::buffer input_buffer = device.create_buffer(CL_MEM_READ_ONLY,
                                                                4 * sizeof(double));
    
    
        kernel_noalias.set_arg(0, output_buffer);
        kernel_noalias.set_arg(1, input_buffer);
    
    
        // main loop
        boost::shared_ptr<workload> next_workload;
        hpx::opencl::work_size<1> dim;
        dim[0].offset = 0;
        while(workqueue->request(&next_workload))
        {
            
            double args[4];
            args[0] = next_workload->origin_x;
            args[1] = next_workload->origin_y;
            args[2] = next_workload->size_x;
            args[3] = next_workload->size_y;
    
            hpx::lcos::shared_future<hpx::opencl::event> ev1 = 
                             input_buffer.enqueue_write(0, 4*sizeof(double), args);
            
            dim[0].size = next_workload->num_pixels;
            hpx::lcos::shared_future<hpx::opencl::event> ev2 = 
                             kernel_noalias.enqueue(dim, ev1);
    
    
            hpx::opencl::event ev3 =
                output_buffer.enqueue_read(
                          0, 3*sizeof(unsigned char)*next_workload->num_pixels, ev2).get();
    
            boost::shared_ptr<std::vector<char>> readdata = ev3.get_data().get();
//            std::cout << readdata->size() << " : " << next_workload->pixeldata.size();
    
            for(size_t i = 0; i < 3 * next_workload->num_pixels; i++)
            {
                next_workload->pixeldata[i] = ((unsigned char*)readdata->data())[i];
            }
            
            // compute workload
            //work(next_workload);
    
            // return workload to work manager workload
            workqueue->deliver(next_workload);
    
        }
    } catch(hpx::exception const& e) {
        
        hpx::cout << "ERROR!" << hpx::endl << e.what() << hpx::endl;
        exit(1);

    }

}

void
mandelbrotworker::work(boost::shared_ptr<workload> next_workload)
{

    
    for(size_t i = 0; i < next_workload->num_pixels; i++)
    {

        double posx = next_workload->origin_x + next_workload->size_x * i /
                                        ((double)next_workload->num_pixels - 1);
        double posy = next_workload->origin_y + next_workload->size_y * i /
                                        ((double)next_workload->num_pixels - 1);

        double betrag_quadrat = 0.0;
        unsigned long iter = 0;
        double x = 0.0;
        double y = 0.0;
        
        double maxiter = 10000;
        
        while(betrag_quadrat <= 10000.0 && iter < maxiter )
        {
            double xt = x * x - y * y + posx;
            double yt = 2 * x * y + posy;
            x = xt;
            y = yt;
            iter = iter + 1;
            betrag_quadrat = x * x + y * y;
        }
        
        double iter_smooth = iter - log(log(betrag_quadrat) / log(4)) / log(2); 
        double val = fmod(0.42*log(iter_smooth) / log(10), 1);
        
        if(iter == maxiter)
        {
            next_workload->pixeldata[3*i+0] = (unsigned char)(0);
            next_workload->pixeldata[3*i+1] = (unsigned char)(0);
            next_workload->pixeldata[3*i+2] = (unsigned char)(0);
        }
        else
        {
            next_workload->pixeldata[3*i+0] = (unsigned char)(200 * val);
            next_workload->pixeldata[3*i+1] = (unsigned char)(200 * val);
            next_workload->pixeldata[3*i+2] = (unsigned char)(200 * val);
        }

    }


}
