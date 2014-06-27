// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MANDELBROT_IMAGE_GENERATOR_H_
#define MANDELBROT_IMAGE_GENERATOR_H_

#include "../../../../opencl.hpp"

#include <boost/shared_ptr.hpp>
#include <vector>

#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/event.hpp>

#include "../work_queue.hpp"
#include "../workload.hpp"
#include "../mandelbrotworker.hpp"

/* 
 * this class is the main observer of the image generation.
 * it gets image queries, which it then splits into subimages and sends it to
 * the calculation queue.
 * it then collects the calculated data and puts it together, to have one finished image.
 */


namespace hpx { namespace opencl { namespace examples { namespace mandelbrot {

struct request;

class maps_image_generator
{

    public:
        // initializes the image generator
        maps_image_generator(size_t img_size_hint_x,
                        size_t img_size_hint_y,
                        size_t num_parallel_kernels,
                        bool verbose,
                        boost::function<boost::shared_ptr<request>(void)>
                            acquire_new_request,
                        std::vector<hpx::opencl::device> devices
                                = std::vector<hpx::opencl::device>());

        // destructor
        ~maps_image_generator();

        // waits for the worker to finish
        void shutdown();
        
        // adds a worker
        void add_worker(hpx::opencl::device & device,
                        size_t num_parallel_kernels);

        // waits for the worker to finish initialization
        void wait_for_startup_finished();
        
    private:
        // the main worker function, runs the main work loop
        static void retrieve_worker_main(
           intptr_t parent_,
           bool verbose);

        // callback for mandelbrotworkers
        bool worker_request_new_work(boost::shared_ptr<workload>* new_work);
        
        // callback for mandelbrotworkers
        void worker_deliver(boost::shared_ptr<workload>& done_work);

        // queries a new image. true on success, false on error.
        bool get_new_image();

        // asynchroneously starts get_new_image().
        void start_getting_new_image();

    // private attributes
    private:
        // for synchronization of workers and retrievers
        hpx::lcos::shared_future<void> retrievers_finished;
        std::vector<boost::shared_ptr<mandelbrotworker>> workers;

        // the actual image data
        typedef std::map<size_t, boost::shared_ptr<request>> 
                    image_request_map;
        image_request_map          images;
        hpx::lcos::local::spinlock images_lock;

        std::atomic_size_t  next_image_id;

        // other stuff
        bool verbose;

        size_t img_size_hint_x;
        size_t img_size_hint_y;

        fifo<boost::shared_ptr<workload>> done_work_queue;
                        
        boost::function<boost::shared_ptr<request>(void)> acquire_new_request;

        hpx::lcos::local::spinlock current_request_lock;
        boost::shared_ptr<request> current_request;
        hpx::lcos::local::condition_variable new_request_available;
        size_t current_request_id;
        size_t current_img_pos;

        volatile bool shutdown_requested;

};

} } } }

#endif

