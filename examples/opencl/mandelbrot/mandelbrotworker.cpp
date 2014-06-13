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

mandelbrotworker::mandelbrotworker(hpx::opencl::device device_,
                                   boost::shared_ptr<work_queue<
                                       boost::shared_ptr<workload>>> workqueue_,
                                   size_t num_workers, bool verbose_)
    : verbose(verbose_),
      device(device_),
      workqueue(workqueue_),
      worker_initialized(boost::shared_ptr<hpx::lcos::local::event>(
                                                new hpx::lcos::local::event()))
{

    // get a unique id
    id = id_counter++;

    // start worker
    worker_finished = hpx::async(&worker_starter,
                                 (intptr_t) this,
                                 num_workers); 

}

mandelbrotworker::~mandelbrotworker()
{

    // wait for the worker to finish
    join();

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
                    intptr_t parent_ptr,
                    hpx::opencl::kernel kernel
           )
{

        // de-serialize parent ptr. dirty hack, but allowed as child WILL
        // be on the same device as parent, and parent will not be 
        // deallocated before child terminated
        mandelbrotworker* parent = (mandelbrotworker*) parent_ptr;


        // counts how much wor has been done
        size_t num_work = 0;

        // create output buffer
        hpx::opencl::buffer output_buffer =
               parent->device.create_buffer(CL_MEM_WRITE_ONLY,
                                            3 * MAX_IMG_WIDTH * sizeof(double));
        // create input buffer
        hpx::opencl::buffer input_buffer = parent->device.create_buffer(
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

        while(parent->workqueue->request(&next_workload))
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
            next_workload->pixeldata = readdata;
            
            // return calculated workload to work manager workload
            parent->workqueue->deliver(next_workload);

            // count number of workloads
            num_work++;
    
        }

        return num_work;

}

void
mandelbrotworker::worker_starter(
           intptr_t parent_ptr,
           size_t num_workers)
{

    // get parent pointer
    mandelbrotworker* parent = (mandelbrotworker*) parent_ptr;


    try{

        std::string device_vendor = parent->device.device_info_to_string(
                                  parent->device.get_device_info(CL_DEVICE_VENDOR));
        std::string device_name = parent->device.device_info_to_string(
                                  parent->device.get_device_info(CL_DEVICE_NAME));
        std::string device_version = parent->device.device_info_to_string(
                                  parent->device.get_device_info(CL_DEVICE_VERSION));

        // print device name
        hpx::cout << "#" << parent->id << ": "
                  << device_vendor << ": "
                  << device_name << " ("
                  << device_version << ")"
                  << hpx::endl;
    
        // build opencl program
        hpx::opencl::program mandelbrot_program =
                     parent->device.create_program_with_source(mandelbrot_kernels);
        if(parent->verbose)
            hpx::cout << "#" << parent->id << ": " << "compiling" << hpx::endl;
        mandelbrot_program.build();
        if(parent->verbose)
            hpx::cout << "#" << parent->id << ": " << "compiling done." << hpx::endl;
    
        
        // start workers
        std::vector<hpx::lcos::shared_future<size_t>> worker_futures;
        for(size_t i = 0; i < num_workers; i++)
        {
         
            // create kernel
            hpx::opencl::kernel kernel = 
                       mandelbrot_program.create_kernel("mandelbrot_alias_8x8");

            // start worker
            hpx::lcos::shared_future<size_t> worker_future = 
                        hpx::async(&worker_main, parent_ptr, kernel);

            // add worker to workerlist
            worker_futures.push_back(worker_future);

        }

        if(parent->verbose)
            hpx::cout << "#" << parent->id << ": " << "workers started!" << hpx::endl;

        // trigger event to start main function.
        // needed for accurate time measurement
        parent->worker_initialized->set();

        // wait for workers to finish
        size_t num_work = 0;
        for(size_t i = 0; i < num_workers; i++)
        {
            // finish worker and get number of computed work packets
            size_t num_work_single = worker_futures[i].get();

            // count total work packets
            num_work += num_work_single;
        }
         
        if(parent->verbose)
        {
            hpx::cout << "#" << parent->id << ": " << "workers finished! ("
                  << num_work << " work packets)" << hpx::endl;
        }

    } catch(hpx::exception const& e) {
        
        // write error message. workaround, should not be done like this in 
        // real application
        hpx::cout << "#" << parent->id << ": " 
                  << "ERROR!" << hpx::endl
                  << hpx::get_error_backtrace(e) << hpx::endl
                  << hpx::diagnostic_information(e) << hpx::endl;

        // kill the process. again, not to be done like this in real application.
        exit(1);

    }

}


