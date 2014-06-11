// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MANDELBROT_MANDELBROTWORKER_H_
#define MANDELBROT_MANDELBROTWORKER_H_

#include "../../../opencl.hpp"

#include <boost/shared_ptr.hpp>

#include "workload.hpp"
#include "work_queue.hpp"

#include <hpx/lcos/local/event.hpp>

/* 
 * a worker.
 * will ask the workqueue for new work until the workqueue finishes.
 * this is the only class that actually uses the hpxcl.
 */
class mandelbrotworker
{

    public:
        // initializes the worker
        mandelbrotworker(hpx::opencl::device device_,
                         boost::shared_ptr<work_queue<
                                      boost::shared_ptr<workload>>> workqueue_,
                         size_t num_workers);

        // waits for the worker to finish
        void join();

        // waits for the worker to finish initialization
        void wait_for_startup_finished();
        
    private:
        // the main worker function, runs the main work loop
        static size_t worker_main(
           boost::shared_ptr<work_queue<boost::shared_ptr<workload>>> workqueue,
           hpx::opencl::device device,
           hpx::opencl::kernel kernel,
           unsigned int id);

        // the startup function, initializes the kernel and starts the workers
        static void worker_starter(
           boost::shared_ptr<work_queue<boost::shared_ptr<workload>>> workqueue,
           hpx::opencl::device device,
           size_t num_workers,
           boost::shared_ptr<hpx::lcos::local::event> worker_initialized,
           unsigned int id);

    // private attributes
    private:
        hpx::lcos::future<void> worker_finished;
        boost::shared_ptr<hpx::lcos::local::event> worker_initialized;

};

#endif

