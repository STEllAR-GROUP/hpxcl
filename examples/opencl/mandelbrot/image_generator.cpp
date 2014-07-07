// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "image_generator.hpp"

#include "../../../opencl.hpp"

#include <hpx/lcos/when_all.hpp>

#include <cmath>

image_generator::
image_generator(size_t img_size_hint_x_,
                size_t img_size_hint_y_,
                size_t num_parallel_kernels,
                bool verbose_,
                std::vector<hpx::opencl::device> devices)
                 : next_image_id(0), verbose(verbose_),
                   img_size_hint_x(img_size_hint_x_),
                   img_size_hint_y(img_size_hint_y_)
{

    // one retrieve worker for every os thread
    size_t num_retrieve_workers = hpx::get_os_thread_count();
   
    // create workqueue
    workqueue = boost::make_shared
                       <work_queue <boost::shared_ptr <workload> > >();
                                
    // initialize worker list
    workers = boost::make_shared 
                       <std::vector <boost::shared_ptr <mandelbrotworker> > >();

    // starting workers
    BOOST_FOREACH(hpx::opencl::device & device, devices)
    {

        // add a worker
        add_worker(device, num_parallel_kernels);

    }
         
    // starting retrievers
    std::vector<hpx::lcos::future<void>> retriever_futures;
    for(size_t i = 0; i < num_retrieve_workers; i++)
    {

        hpx::lcos::future<void> retriever_future = 
                    hpx::async(retrieve_worker_main,
                               (intptr_t) this,
                               verbose);

        retriever_futures.push_back(std::move(retriever_future));

    }

    // combining all retrievers into one future
    retrievers_finished = hpx::when_all(retriever_futures).share();
    
}


image_generator::
~image_generator()
{

    // wait for work to get finished
    shutdown();

}

void
image_generator::
add_worker(hpx::opencl::device & device, size_t num_parallel_kernels)
{

        // create request callback function for worker
        boost::function<bool(boost::shared_ptr<workload>*)> request_new_work = 
               boost::bind(&work_queue<boost::shared_ptr<workload>>::request,
                           &(*workqueue),
                           _1); 

        // create deliver callback function for worker
        boost::function<void(boost::shared_ptr<workload>&)> deliver_done_work = 
               boost::bind(&work_queue<boost::shared_ptr<workload>>::deliver,
                           &(*workqueue),
                           _1); 


        // create worker
        boost::shared_ptr<mandelbrotworker> worker = 
            boost::make_shared<mandelbrotworker>
                                (device,
                                 num_parallel_kernels,
                                 request_new_work,
                                 deliver_done_work,
                                 verbose,
                                 img_size_hint_x,
                                 img_size_hint_y);

        // add worker to workerlist
        workers->push_back(worker);

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
    retrievers_finished.wait();

}

void
image_generator::
retrieve_worker_main(intptr_t parent_, bool verbose)
{

    // get parent pointer
    image_generator* parent = (image_generator*) parent_;

    // represents done workload
    boost::shared_ptr<workload> done_workload;

    // main loop
    if(verbose) hpx::cout << "entering retrieve worker main loop ..." << hpx::endl;
    while(parent->workqueue->retrieve_finished_work(&done_workload))
    {

        // retrieve id of associated image
        size_t img_id = done_workload->img_id;

        // image data
        boost::shared_ptr<std::vector<char>> img_data;

        // image countdown
        boost::shared_ptr<std::atomic<size_t>> img_countdown;

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

        // copy data to img_data
        if(img_data)
        {
            size_t start_x = done_workload->pos_in_img_x;
            size_t start_y = done_workload->pos_in_img_y;
            size_t size_x = done_workload->num_pixels_x;
            size_t size_y = done_workload->num_pixels_y;
            size_t line_offset = done_workload->line_offset;
            for(size_t y = 0; y < size_y; y++)
            {
                for(size_t x = 0; x < size_x; x++)
                {
                    (*img_data)[((y + start_y) * line_offset + (x + start_x)) * 3 + 0] =
                        (*(done_workload->pixeldata))[(y * size_x + x) * 3 + 0];  
                    (*img_data)[((y + start_y) * line_offset + (x + start_x)) * 3 + 1] =
                        (*(done_workload->pixeldata))[(y * size_x + x) * 3 + 1];  
                    (*img_data)[((y + start_y) * line_offset + (x + start_x)) * 3 + 2] =
                        (*(done_workload->pixeldata))[(y * size_x + x) * 3 + 2];  
                }
            }
        }

        // decrease the number of work packets left
        size_t current_img_countdown = --(*img_countdown);
        if(verbose) hpx::cout << "retrieved workload " << current_img_countdown << ": "
                              << done_workload->pos_in_img_x
                              << ":" 
                              << done_workload->pos_in_img_y
                              << hpx::endl;


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
              size_t img_height)
{

    return compute_image(posx, posy, zoom, rotation,
                         img_width, img_height,
                         false, img_width, 1);

}

hpx::lcos::future<boost::shared_ptr<std::vector<char>>>
image_generator::
compute_image(double posx,
              double posy,
              double zoom,
              double rotation,
              size_t img_width,
              size_t img_height,
              bool benchmark,
              size_t tile_width,
              size_t tile_height)
{

    // calculate image id
    size_t img_id = next_image_id++;

    // calculate aspect ratio
    double aspect_ratio = (double) img_width / (double) img_height;

    // calculate size of diagonale
    //double size_diag = exp2(-zoom) * 4.0;
    double size_diag = 4.0 / zoom;
    
    // calculate width and height
    double size_y = size_diag / sqrt( 1 + aspect_ratio * aspect_ratio );
    double size_x = aspect_ratio * size_y;

    // calculate horizontal stepwidth
    double hor_pixdist_nonrot = size_x / (img_width - 1);
    double hor_pixdist_x = cos(rotation) * hor_pixdist_nonrot;
    double hor_pixdist_y = sin(rotation) * hor_pixdist_nonrot;

    // calculate vertical stepwidth
    double vert_pixdist_nonrot = - size_y / (img_height - 1);
    double vert_pixdist_x = - sin(rotation) * vert_pixdist_nonrot;
    double vert_pixdist_y = cos(rotation) * vert_pixdist_nonrot;


    // calculate top left coords
    double topleft_x = posx - hor_pixdist_x * ( img_width / 2.0 + 0.5 ) 
                            - vert_pixdist_x * ( img_height / 2.0 + 0.5 );
    double topleft_y = posy - hor_pixdist_y * ( img_width / 2.0 + 0.5 ) 
                            - vert_pixdist_y * ( img_height / 2.0 + 0.5 );

    // calculate number of tiles
    BOOST_ASSERT(img_width % tile_width == 0 && img_height % tile_height == 0);
    size_t num_tiles_x = img_width / tile_width;
    size_t num_tiles_y = img_height / tile_height;

    if(verbose){
        hpx::cout << "image data" << hpx::endl
              << "topleft:       " << topleft_x << ":" << topleft_y << hpx::endl
              << "img_dims:      " << img_width << ":" << img_height << hpx::endl
              << "pos:           " << posx << ":" << posy << hpx::endl
              << "size:          " << size_x << ":" << size_y << hpx::endl
              << "hor_pixdist:   " << hor_pixdist_x << ":" << hor_pixdist_y << hpx::endl
              << "vert_pixdist:  " << vert_pixdist_x << ":" << vert_pixdist_y << hpx::endl
              << "num_tiles:     " << num_tiles_x << ":" << num_tiles_y << hpx::endl;
    }

    // create data array to hold finished image, if we are not in benchmark mode
    boost::shared_ptr<std::vector<char>> img_data;
    if(!benchmark)
    img_data = boost::make_shared <std::vector <char> >
                                    (img_width * img_height * 3 * sizeof(char));

    // create a new countdown variable
    boost::shared_ptr<std::atomic<size_t>> img_countdown = 
              boost::make_shared<std::atomic<size_t>>(num_tiles_x * num_tiles_y);

    // create a new ready event lock
    boost::shared_ptr<hpx::lcos::local::event> img_ready =
              boost::make_shared<hpx::lcos::local::event>();

    // add the created variables to their lists
    {
        boost::lock_guard<hpx::lcos::local::spinlock>
        lock(images_lock);

        // do not add data in benchmark mode
        if(!benchmark)
        images.insert(std::pair<size_t, boost::shared_ptr<std::vector<char>>>
                                                            (img_id, img_data));
        images_countdown.insert(std::pair<size_t,
                                boost::shared_ptr<std::atomic<size_t>>>
                                (img_id, img_countdown));
        images_ready.insert(std::pair<size_t,
                            boost::shared_ptr<hpx::lcos::local::event>>
                            (img_id, img_ready));
    }


    // add the workloads to queue
    if (verbose) hpx::cout << "Adding workloads to queue ..." << hpx::endl;
    for(size_t y = 0; y < img_height; y += tile_height)
    {
        for(size_t x = 0; x < img_width; x += tile_width)
        {
            if (verbose) hpx::cout << "\tAdding workload " << x << ":" << y << " ..." << hpx::endl;
            
            // calculate position of current work packet
            double workpacket_pos_x = topleft_x + vert_pixdist_x * y + hor_pixdist_x * x;
            double workpacket_pos_y = topleft_y + vert_pixdist_y * y + hor_pixdist_y * x;
            // add workload
            boost::shared_ptr<workload> row =
                       boost::make_shared<workload>(tile_width,
                                                    tile_height,
                                                    workpacket_pos_x,
                                                    workpacket_pos_y,
                                                    hor_pixdist_x, 
                                                    hor_pixdist_y,
                                                    vert_pixdist_x,
                                                    vert_pixdist_y,
                                                    img_id,
                                                    x,
                                                    y,
                                                    img_width);
            workqueue->add_work(row);
        }
    }
        
    // return the future to the finished image
    return hpx::async(wait_for_image_finished, img_ready, img_data);

}
