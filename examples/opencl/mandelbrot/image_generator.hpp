// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MANDELBROT_IMAGE_GENERATOR_H_
#define MANDELBROT_IMAGE_GENERATOR_H_

#include "../../../opencl.hpp"

#include <boost/shared_ptr.hpp>
#include <vector>

#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/lcos/local/event.hpp>

#include "work_queue.hpp"
#include "workload.hpp"
#include "mandelbrotworker.hpp"
#include <atomic>

/* 
 * this class is the main observer of the image generation.
 * it gets image queries, which it then splits into subimages and sends it to
 * the calculation queue.
 * it then collects the calculated data and puts it together, to have one finished image.
 */
class image_generator
{

    public:
        // initializes the image generator
        image_generator(size_t img_size_hint_x,
                        size_t img_size_hint_y,
                        size_t num_parallel_kernels,
                        bool verbose,
                        std::vector<hpx::opencl::device> devices
                                = std::vector<hpx::opencl::device>());

        // destructor
        ~image_generator();

        // waits for the worker to finish
        void shutdown();
        
        // adds a worker
        void add_worker(hpx::opencl::device & device,
                        size_t num_parallel_kernels);

        // waits for the worker to finish initialization
        void wait_for_startup_finished();
        
        // computes an image
        hpx::lcos::future<boost::shared_ptr<std::vector<char>>>
        compute_image(double pos_x,
                      double pos_y,
                      double zoom,
                      double rotation,
                      size_t img_width,
                      size_t img_height);

        // computes an image, enhanced version
        hpx::lcos::future<boost::shared_ptr<std::vector<char>>>
        compute_image(double pos_x,
                      double pos_y,
                      double zoom,
                      double rotation,
                      size_t img_width,
                      size_t img_height,
                      bool benchmark, // purges output
                      size_t tile_width,
                      size_t tile_height);

    private:
        // the main worker function, runs the main work loop
        static void retrieve_worker_main(
           intptr_t parent_,
           bool verbose);


    // private attributes
    private:
        hpx::lcos::shared_future<void> retrievers_finished;
        boost::shared_ptr<work_queue<boost::shared_ptr<workload>>> workqueue;
        boost::shared_ptr<std::vector<boost::shared_ptr<mandelbrotworker>>> workers;
        hpx::lcos::local::spinlock images_lock;

        typedef std::map<size_t, boost::shared_ptr<std::vector<char>>> 
                    image_data_map;
        typedef std::map<size_t, boost::shared_ptr<std::atomic<size_t>>>
                    image_countdown_map;
        typedef std::map<size_t, boost::shared_ptr<hpx::lcos::local::event>>
                    image_ready_map;
        image_data_map      images;
        image_countdown_map images_countdown;
        image_ready_map     images_ready;

        std::atomic<size_t>  next_image_id;
        bool verbose;

        size_t img_size_hint_x;
        size_t img_size_hint_y;

};

#endif

