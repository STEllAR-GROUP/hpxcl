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
                         size_t num_workers,
                         bool verbose, size_t workpacket_size_hint);

        // waits for the worker to finish
        void join();

        // waits for the worker to finish initialization
        void wait_for_startup_finished();
        
        // destructor, basically waits for the worker to finish
        ~mandelbrotworker();

    private:
        // the main worker function, runs the main work loop
        static size_t worker_main(
           intptr_t parent_,
           hpx::opencl::kernel kernel,
           size_t workpacket_size_hint);

        // the startup function, initializes the kernel and starts the workers
        static void worker_starter(
           intptr_t parent_,
           size_t num_workers,
           size_t workpacket_size_hint
           );

    // private attributes
    private:
        const bool verbose;
        const unsigned int id;
        hpx::opencl::device device;
        boost::shared_ptr<work_queue<boost::shared_ptr<workload>>> workqueue;
        hpx::lcos::shared_future<void> worker_finished;
        boost::shared_ptr<hpx::lcos::local::event> worker_initialized;
};

#endif

