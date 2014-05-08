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
                                      boost::shared_ptr<workload>>> workqueue_);
        // waits for the worker to finish
        void join();
        
    private:
        // the main worker function, runs the main work loop
        static void worker_main(
           boost::shared_ptr<work_queue<boost::shared_ptr<workload>>> workqueue,
           hpx::opencl::device device);

        // calculates the workload
        static void work(boost::shared_ptr<workload>);

    // private attributes
    private:
        boost::shared_ptr<work_queue<boost::shared_ptr<workload>>> workqueue;
        hpx::opencl::device device;
        hpx::lcos::future<void> worker_finished;


};

#endif

