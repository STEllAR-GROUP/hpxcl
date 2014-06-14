// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "image_generator.hpp"

#include "../../../opencl.hpp"

#include <hpx/lcos/when_all.hpp>


image_generator::
image_generator(boost::shared_ptr<std::vector<hpx::opencl::device>> devices,
                size_t img_size_hint,
                size_t num_parallel_kernels,
                bool verbose) : next_image_id(0)
{

    // one retrieve worker for every os thread
    size_t num_retrieve_workers = hpx::get_os_thread_count();
   
    // create workqueue
    workqueue = boost::shared_ptr<work_queue<
                        boost::shared_ptr<workload>>>
                (new work_queue<boost::shared_ptr<workload>>());
                                
    // initialize worker list
    workers = boost::shared_ptr<std::vector<
                                boost::shared_ptr<mandelbrotworker>>>
             (new std::vector<boost::shared_ptr<mandelbrotworker>>());

    // starting workers
    BOOST_FOREACH(hpx::opencl::device & device, *devices)
    {

        // create worker
        boost::shared_ptr<mandelbrotworker> worker(
            new mandelbrotworker(device,
                                 workqueue,
                                 num_parallel_kernels,
                                 verbose,
                                 img_size_hint));

        // add worker to workerlist
        workers->push_back(worker);

    }
         
    // starting retrievers
    std::vector<hpx::lcos::shared_future<void>> retriever_futures;
    for(size_t i = 0; i < num_retrieve_workers; i++)
    {

        hpx::lcos::shared_future<void> retriever_future = 
                    hpx::async(retrieve_worker_main,
                               (intptr_t) this);

        retriever_futures.push_back(retriever_future);

    }

    // combining all retrievers into one future
    retrievers_finished = hpx::when_all(retriever_futures);
    
}


image_generator::
~image_generator()
{

    // wait for work to get finished
    shutdown();

}

void
image_generator::
wait_for_startup_finished()
{

    // wait for all workers to finish startup
    BOOST_FOREACH(boost::shared_ptr<mandelbrotworker> & worker, *workers)
    {

        worker->wait_for_startup_finished();

    }


}

void 
image_generator::
shutdown()
{

    // end workqueue
    workqueue->finish();

    // wait for retrievers to finish
    retrievers_finished.get();

}

void
image_generator::
retrieve_worker_main(intptr_t parent_)
{

    // get parent pointer
    image_generator* parent = (image_generator*) parent_;

    // represents done workload
    boost::shared_ptr<workload> done_workload;

    // main loop
    while(parent->workqueue->retrieve_finished_work(&done_workload))
    {

        // retrieve id of associated image
        size_t img_id = done_workload->img_id;

        // image data
        boost::shared_ptr<std::vector<char>> img_data;

        // image countdown
        boost::shared_ptr<std::atomic_size_t> img_countdown;

        // image event lock
        boost::shared_ptr<hpx::lcos::local::event> img_ready;

        // retrieve image pointers
        {
            // lock
            boost::lock_guard<hpx::lcos::local::spinlock>
            lock(parent->images_lock);
            
            // retrieve image data
            image_data_map::iterator data_iterator = parent->images.find(img_id);
            // leave as null pointer if no data exists.
            // this indicates benchmark mode.
            if(data_iterator != parent->images.end())
                img_data = data_iterator->second;

            // retrieve image countdown
            BOOST_ASSERT(parent->images_countdown.find(img_id)
                                != parent->images_countdown.end());
            img_countdown = parent->images_countdown[img_id];

            // retrieve image event lock
            BOOST_ASSERT(parent->images_ready.find(img_id)
                                != parent->images_ready.end());
            img_ready = parent->images_ready[img_id];
        }

        // TODO copy data to img_data

        // decrease the number of work packets left
        size_t current_img_countdown = --(*img_countdown);

        // if no work packet left (img finished), then:
        if(current_img_countdown == 0)
        {
            // set the image ready event lock
            img_ready->set(); 

            // lock the data lists
            boost::lock_guard<hpx::lcos::local::spinlock>
            lock(parent->images_lock);

            // remove image data.
            // data will still be available for waiting image thread,
            // as it is a shared_ptr.
            image_data_map::iterator data_it = parent->images.find(img_id);
            if(data_it != parent->images.end())
                parent->images.erase(data_it);

            // remove countdown variable
            image_countdown_map::iterator countdown_it = 
                                parent->images_countdown.find(img_id); 
            parent->images_countdown.erase(countdown_it);

            // remove event lock
            image_ready_map::iterator ready_it = 
                                parent->images_ready.find(img_id);
            parent->images_ready.erase(ready_it);
        }

    }

}

// waits until event lock triggered, then returns data
boost::shared_ptr<std::vector<char>>
wait_for_image_finished(boost::shared_ptr<hpx::lcos::local::event> img_ready,
                        boost::shared_ptr<std::vector<char>> img_data)
{

    // wait for the event lock to trigger
    img_ready->wait();

    // return the image data
    return img_data;

}

hpx::lcos::future<boost::shared_ptr<std::vector<char>>>
image_generator::
compute_image(double posx,
              double posy,
              double zoom,
              double rotation,
              size_t img_width,
              size_t img_height,
              bool benchmark)
{

    // calculate image id
    size_t img_id = next_image_id++;

    // create data array to hold finished image, if we are not in benchmark mode
    boost::shared_ptr<std::vector<char>> img_data;
    if(!benchmark)
    img_data = boost::shared_ptr<std::vector<char>>(
                new std::vector<char>(img_width * img_height * 3 * sizeof(char)));

    // create a new countdown variable
    boost::shared_ptr<std::atomic_size_t> img_countdown (new std::atomic_size_t(img_height));

    // create a new ready event lock
    boost::shared_ptr<hpx::lcos::local::event> img_ready (new hpx::lcos::local::event());

    // add the created variables to their lists
    {
        boost::lock_guard<hpx::lcos::local::spinlock>
        lock(images_lock);

        // do not add data in benchmark mode
        if(!benchmark)
        images.insert(std::pair<size_t, boost::shared_ptr<std::vector<char>>>
                                                            (img_id, img_data));
        images_countdown.insert(std::pair<size_t,
                                boost::shared_ptr<std::atomic_size_t>>
                                (img_id, img_countdown));
        images_ready.insert(std::pair<size_t,
                            boost::shared_ptr<hpx::lcos::local::event>>
                            (img_id, img_ready));
    }

    // temporal, will be replaced
    double left = -0.743643887062801003142;
    double right = -0.743643887011500996858;
    double top = 0.131825904224567502357;
    double bottom = 0.131825904186092497643;

    // add the workloads to queue
    for(size_t i = 0; i < img_height; i++)
    {
        // hpx::cout << "adding line " << i << " ..." << hpx::endl;
        boost::shared_ptr<workload> row(
               new workload(img_width, left, top - (top - bottom)*i/(img_height - 1), right-left, 0.0, 0, i));
        workqueue->add_work(row);

    }
        
    // return the future to the finished image
    return hpx::async(wait_for_image_finished, img_ready, img_data);

}
