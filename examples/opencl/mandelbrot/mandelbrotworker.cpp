// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mandelbrotworker.hpp"
#include "mandelbrotkernel.hpp"
#include "mandelbrotworker_buffermanager.hpp"

#include <cmath>
#include <boost/atomic.hpp>

#define MAX_IMG_WIDTH 30000

static boost::atomic<unsigned int> id_counter((unsigned int)0);

mandelbrotworker::mandelbrotworker(
                         hpx::opencl::device device_,
                         size_t num_workers, 
                         boost::function<bool(boost::shared_ptr<workload>*)>
                         request_new_work_,
                         boost::function<void(boost::shared_ptr<workload> &)>
                         deliver_done_work_,
                         bool verbose_,
                         size_t workpacket_size_hint_x,
                         size_t workpacket_size_hint_y)
    : verbose(verbose_),
      id(id_counter++),
      device(device_),
      worker_initialized(boost::make_shared<hpx::lcos::local::event>()),
      request_new_work(request_new_work_),
      deliver_done_work(deliver_done_work_)
{

    // start worker
    worker_finished = hpx::async(&mandelbrotworker::worker_starter,
                                 this,
                                 num_workers,
                                 workpacket_size_hint_x,
                                 workpacket_size_hint_y); 

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
    hpx::shared_future<void> tmp = worker_finished;
    tmp.wait();
    
}

void
mandelbrotworker::wait_for_startup_finished()
{

    // waits until the worker_starter triggers this event
    worker_initialized->wait();

}

#define KERNEL_INPUT_ARGUMENT_COUNT 6
size_t
mandelbrotworker::worker_main(
                    hpx::opencl::kernel precalc_kernel,
                    hpx::opencl::kernel kernel,
                    size_t workpacket_size_hint_x,
                    size_t workpacket_size_hint_y
           )
{

        // setup device memory management.
        // initialize default buffer with size of numpixels * 3 (rgb) * sizeof(double)
        mandelbrotworker_buffermanager buffermanager(
                                        device,
                                        workpacket_size_hint_x 
                                        * workpacket_size_hint_y 
                                        * 3 * sizeof(char),
                                        verbose,
                                        CL_MEM_WRITE_ONLY); 

        // initialize buffermanager for precalc buffer
        mandelbrotworker_buffermanager precalc_buffermanager(
                                        device,
                                        (workpacket_size_hint_x + 2)
                                        * (workpacket_size_hint_y + 2)
                                        * sizeof(char),
                                        verbose,
                                        CL_MEM_READ_WRITE); 

        // counts how much work has been done
        size_t num_work = 0;

        // attach output buffer
        size_t current_buffer_size = workpacket_size_hint_x
                                     * workpacket_size_hint_y
                                     * 3 * sizeof(char);
        hpx::opencl::buffer output_buffer = buffermanager.get_buffer( 
                                                    current_buffer_size );

        // attach precalc buffer
        size_t current_precalc_size = (workpacket_size_hint_x + 2)
                                      * (workpacket_size_hint_y + 2)
                                      * sizeof(char);
        hpx::opencl::buffer precalc_buffer = precalc_buffermanager.get_buffer(
                                                    current_precalc_size );

        // create input buffer
        hpx::opencl::buffer input_buffer = device.create_buffer(
                                                     CL_MEM_READ_ONLY,
                                                     KERNEL_INPUT_ARGUMENT_COUNT * sizeof(double));
    
        // connect buffers to kernel 
        kernel.set_arg(0, precalc_buffer);
        kernel.set_arg(1, output_buffer);
        kernel.set_arg(2, input_buffer);
    
        // connect buffers to precalc kernel 
        precalc_kernel.set_arg(0, precalc_buffer);
        precalc_kernel.set_arg(1, input_buffer);
    
    
        // main loop
        boost::shared_ptr<workload> next_workload;
        hpx::opencl::work_size<2> dim;
        dim[0].offset = 0;
        dim[1].offset = 0;
        dim[0].local_size = 8;
        dim[1].local_size = 8;

        hpx::opencl::work_size<2> precalc_dim;
        precalc_dim[0].offset = 0;
        precalc_dim[1].offset = 0;

        while(request_new_work(&next_workload))
        {
            
            // calculate output buffer size
            size_t needed_buffer_size = next_workload->num_pixels_x
                                        * next_workload->num_pixels_y
                                        * 3
                                        * sizeof(char);


            // change output buffer if needed buffersize changed
            if (current_buffer_size != needed_buffer_size)
            {
                // query new buffer
                output_buffer = buffermanager.get_buffer( needed_buffer_size );

                // attach new buffer
                kernel.set_arg(1, output_buffer);

                // update current buffer size
                current_buffer_size = needed_buffer_size;
            }
                                        
            // calculate precalc buffer size
            size_t needed_precalc_size = (next_workload->num_pixels_x + 2)
                                        * (next_workload->num_pixels_y + 2)
                                        * sizeof(char);

            // change precalc buffer if needed precalcsize changed
            if (current_precalc_size != needed_precalc_size)
            {
                // query new buffer
                precalc_buffer = precalc_buffermanager.get_buffer(
                                                         needed_precalc_size );

                // attach new buffer
                auto fut = kernel.set_arg_async(0, precalc_buffer);
                precalc_kernel.set_arg(0, precalc_buffer);
                fut.get();

                // update current buffer size
                current_precalc_size = needed_precalc_size;
            }
 
            // read calculation dimensions
            double args[KERNEL_INPUT_ARGUMENT_COUNT];
            args[0] = next_workload->topleft_x;
            args[1] = next_workload->topleft_y;
            args[2] = next_workload->hor_pixdist_x;
            args[3] = next_workload->hor_pixdist_y;
            args[4] = next_workload->vert_pixdist_x;
            args[5] = next_workload->vert_pixdist_y;
            typedef hpx::serialization::serialize_buffer<double> double_buffer_type;
            double_buffer_type args_buf( args, KERNEL_INPUT_ARGUMENT_COUNT,
                                         double_buffer_type::init_mode::reference );
    
            // send calculation dimensions to gpu
            auto ev1 = input_buffer.enqueue_write(0, args_buf);

            // run precalculation
            precalc_dim[0].size = next_workload->num_pixels_x + 2;
            precalc_dim[1].size = next_workload->num_pixels_y + 2;
            auto ev2 = precalc_kernel.enqueue(precalc_dim, ev1);
            
             // run calculation
            dim[0].size = next_workload->num_pixels_x * 8;
            dim[1].size = next_workload->num_pixels_y * 8;
            auto ev3 = kernel.enqueue(dim, ev2);
    
            // query calculation result 
            auto ev4 = output_buffer.enqueue_read(0, current_buffer_size, ev3);
    
            // wait for calculation result to arrive
            hpx::serialization::serialize_buffer<char> readdata = ev4.get();
    
            // copy calculation result to output buffer
            next_workload->pixeldata = readdata;
            
            // return calculated workload to work manager workload
            deliver_done_work(next_workload);

            // count number of workloads
            num_work++;
    
        }

        return num_work;

}

void
mandelbrotworker::worker_starter(
           size_t num_workers,
           size_t workpacket_size_hint_x,
           size_t workpacket_size_hint_y)
{


    try{

        std::string device_vendor = device.get_device_info<CL_DEVICE_VENDOR>().get();
        std::string device_name = device.get_device_info<CL_DEVICE_NAME>().get();
        std::string device_version = device.get_device_info<CL_DEVICE_VERSION>().get();

        // print device name
        hpx::cerr << "#" << id << ": "
                  << device_vendor << ": "
                  << device_name << " ("
                  << device_version << ")"
                  << hpx::endl;
    
        // build opencl program
        typedef hpx::serialization::serialize_buffer<char> char_buffer_type;
        char_buffer_type mandelbrotkernel_buf
            ( mandelbrotkernel_cl, mandelbrotkernel_cl_len,
              char_buffer_type::init_mode::reference );
                                                
        hpx::opencl::program mandelbrot_program =
                     device.create_program_with_source(mandelbrotkernel_buf);
        if(verbose)
            hpx::cout << "#" << id << ": " << "compiling" << hpx::endl;
        mandelbrot_program.build();
        if(verbose)
            hpx::cout << "#" << id << ": " << "compiling done." << hpx::endl;
    
        
        // start workers
        std::vector<hpx::lcos::future<size_t>> worker_futures;
        for(size_t i = 0; i < num_workers; i++)
        {
         
            // create kernel
            hpx::opencl::kernel kernel = 
                       mandelbrot_program.create_kernel("mandelbrot_alias_8x8");

            // create precalc kernel
            hpx::opencl::kernel precalc_kernel = 
                       mandelbrot_program.create_kernel("precompute_mandelbrot");

            // start worker
            hpx::lcos::future<size_t> worker_future = 
                                      hpx::async(&mandelbrotworker::worker_main,
                                                 this,
                                                 precalc_kernel,
                                                 kernel,
                                                 workpacket_size_hint_x,
                                                 workpacket_size_hint_y);

            // add worker to workerlist
            worker_futures.push_back(std::move(worker_future));

        }

        if(verbose)
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

            // count total work packets
            num_work += num_work_single;
        }
         
        if(verbose)
        {
            hpx::cout << "#" << id << ": " << "workers finished! ("
                  << num_work << " work packets)" << hpx::endl;
        }

    } catch(hpx::exception const& e) {
        
        // write error message. workaround, should not be done like this in 
        // real application
        hpx::cout << "#" << id << ": " 
                  << "ERROR!" << hpx::endl
                  << hpx::get_error_backtrace(e) << hpx::endl
                  << hpx::diagnostic_information(e) << hpx::endl;

        // kill the process. again, not to be done like this in real application.
        exit(1);

    }

}


