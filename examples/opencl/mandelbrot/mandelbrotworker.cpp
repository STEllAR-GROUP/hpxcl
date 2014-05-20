// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mandelbrotworker.hpp"
#include "mandelbrotkernel.hpp"

#include <cmath>
#include <atomic>

#define MAX_IMG_WIDTH 30000

static std::atomic_uint id_counter(0);

mandelbrotworker::mandelbrotworker(hpx::opencl::device device,
                                   boost::shared_ptr<work_queue<
                                       boost::shared_ptr<workload>>> workqueue,
                                   size_t num_workers)
    : worker_initialized(boost::shared_ptr<hpx::lcos::local::event>(
                                                new hpx::lcos::local::event()))
{

    // get a unique id
    unsigned int id = id_counter++;

    // start worker
    worker_finished = hpx::async(&worker_starter,
                                 workqueue,
                                 device,
                                 num_workers,
                                 worker_initialized, id); 

}


void
mandelbrotworker::join()
{

    // wait for worker to finish
    worker_finished.get();
    
}

void
mandelbrotworker::wait_for_startup_finished()
{

    // waits until the worker_starter triggers this event
    worker_initialized->wait();

}

size_t
mandelbrotworker::worker_main(
           boost::shared_ptr<work_queue<boost::shared_ptr<workload>>> workqueue,
           hpx::opencl::device device,
           hpx::opencl::kernel kernel,
           unsigned int id)
{

        // counts how much wor has been done
        size_t num_work = 0;

        // create output buffer
        hpx::opencl::buffer output_buffer =
                       device.create_buffer(CL_MEM_WRITE_ONLY,
                                            3 * MAX_IMG_WIDTH * sizeof(double));
        // create input buffer
        hpx::opencl::buffer input_buffer = device.create_buffer(
                                                  CL_MEM_READ_ONLY,
                                                  4 * sizeof(double));
    
        // connect buffers to kernel 
        kernel.set_arg(0, output_buffer);
        kernel.set_arg(1, input_buffer);
    
    
        // main loop
        boost::shared_ptr<workload> next_workload;
        hpx::opencl::work_size<2> dim;
        dim[0].offset = 0;
        dim[1].offset = 0;
        dim[1].size = 8;
        dim[0].local_size = 8;
        dim[1].local_size = 8;

        while(workqueue->request(&next_workload))
        {
            
            // check for maximum workload size
            if(next_workload->num_pixels > MAX_IMG_WIDTH)
            {
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                          "mandelbrotworker::worker_main()",
                          "ERROR: workload size is larger than MAX_IMG_WIDTH!");
            }

            // read calculation dimensions
            double args[4];
            args[0] = next_workload->origin_x;
            args[1] = next_workload->origin_y;
            args[2] = next_workload->size_x;
            args[3] = next_workload->size_y;
    
            // send calculation dimensions to gpu
            hpx::lcos::shared_future<hpx::opencl::event> ev1 = 
                          input_buffer.enqueue_write(0, 4*sizeof(double), args);
            
            // run calculation
            dim[0].size = next_workload->num_pixels * 8;
            hpx::lcos::shared_future<hpx::opencl::event> ev2 = 
                                                       kernel.enqueue(dim, ev1);
    
            // query calculation result 
            hpx::opencl::event ev3 =
                output_buffer.enqueue_read(
                          0, 3*sizeof(unsigned char)*next_workload->num_pixels, ev2).get();
    
            // wait for calculation result to arrive
            boost::shared_ptr<std::vector<char>> readdata = ev3.get_data().get();
    
            // copy calculation result to output buffer
            for(size_t i = 0; i < 3 * next_workload->num_pixels; i++)
            {
                next_workload->pixeldata[i] = ((unsigned char*)readdata->data())[i];
            }
            
            // return calculated workload to work manager workload
            workqueue->deliver(next_workload);

            // count number of workloads
            num_work++;
    
        }

        return num_work;

}

void
mandelbrotworker::worker_starter(
           boost::shared_ptr<work_queue<boost::shared_ptr<workload>>> workqueue,
           hpx::opencl::device device,
           size_t num_workers,
           boost::shared_ptr<hpx::lcos::local::event> worker_initialized,
           unsigned int id)
{
    try{
        // print device name
        hpx::cout << "#" << id << ": "
                  << device.device_info_to_string(
                                  device.get_device_info(CL_DEVICE_VENDOR))
                  << ": "
                  << device.device_info_to_string(
                                  device.get_device_info(CL_DEVICE_NAME))
                  << " ("
                  << device.device_info_to_string(
                                  device.get_device_info(CL_DEVICE_VERSION))
                  << ")"
                  << hpx::endl;
    
        // build opencl program
        hpx::opencl::program mandelbrot_program =
                          device.create_program_with_source(mandelbrot_kernels);
        hpx::cout << "#" << id << ": " << "compiling" << hpx::endl;
        mandelbrot_program.build();
        hpx::cout << "#" << id << ": " << "compiling done." << hpx::endl;
    
        
        // start workers
        std::vector<hpx::lcos::shared_future<size_t>> worker_futures;
        for(size_t i = 0; i < num_workers; i++)
        {
         
            // create kernel
            hpx::opencl::kernel kernel = 
                       mandelbrot_program.create_kernel("mandelbrot_alias_8x8");

            // start worker
            hpx::lcos::shared_future<size_t> worker_future = 
                        hpx::async(&worker_main, workqueue, device, kernel, id);

            // add worker to workerlist
            worker_futures.push_back(worker_future);

        }

        hpx::cout << "#" << id << ": " << "workers started!" << hpx::endl;

        // trigger event to start main function.
        // needed for accurate time measurement
        worker_initialized->set();

        // wait for workers to finish
        size_t num_work = 0;
        for(size_t i = 0; i < num_workers; i++)
        {
            // finish worker and get number of computed work packets
            size_t num_work_single = worker_futures[i].get();

            // display work packet count
            hpx::cout << "#" << id << ": " << "worker " << i 
                      << ": " << num_work_single << " work packets"
                      << hpx::endl;

            // count total work packets
            num_work += num_work_single;
        }
         
        hpx::cout << "#" << id << ": " << "workers finished! ("
                  << num_work << " work packets)" << hpx::endl;

    } catch(hpx::exception const& e) {
        
        // write error message. workaround, should not be done like this in 
        // real application
        hpx::cout << "#" << id << ": " 
                  << "ERROR!" << hpx::endl << e.what() << hpx::endl;

        // kill the process. again, not to be done like this in real application.
        exit(1);

    }

}


