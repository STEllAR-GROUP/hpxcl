// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "maps_image_generator.hpp"

#include "../../../../opencl.hpp"

#include <hpx/lcos/when_all.hpp>

#include <cmath>

#include "../pngwriter.hpp"
#include "requesthandler.hpp"


using namespace hpx::opencl::examples::mandelbrot;

maps_image_generator::
maps_image_generator(size_t img_size_hint_x_,
                size_t img_size_hint_y_,
                size_t num_parallel_kernels,
                bool verbose_,
                boost::function<boost::shared_ptr<request>(void)>
                    acquire_new_request_,
                std::vector<hpx::opencl::device> devices)
                 : next_image_id(0), verbose(verbose_),
                   img_size_hint_x(img_size_hint_x_),
                   img_size_hint_y(img_size_hint_y_),
                   acquire_new_request(acquire_new_request_),
                   shutdown_requested(false)
{

    // one retrieve worker for every os thread
    size_t num_retrieve_workers = hpx::get_os_thread_count();
   
    // starting workers
    for( auto& device : devices)
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
    
    // start the first image fetch
    start_getting_new_image();
}


maps_image_generator::
~maps_image_generator()
{

    // wait for work to get finished
    shutdown();

}

// current_request_lock must be locked BEFORE entering this function!
void 
maps_image_generator::
dispose_current_request_if_invalid()
{

    if(current_request)
    {
        if(!current_request->stillValid())
        {
            current_request->abort();
            
            {                                           
                // lock the data list
                boost::lock_guard<hpx::lcos::local::spinlock>
                lock2(images_lock);
        
                // insert new request to images list
                images.erase(current_request_id);
            }
    
            current_request = boost::shared_ptr<request>();
    
            start_getting_new_image();
        }
    }


}

bool
maps_image_generator::
worker_request_new_work(boost::shared_ptr<workload>* new_work)
{

    // lock
    boost::lock_guard<hpx::lcos::local::spinlock>
    lock(current_request_lock);

    if(shutdown_requested) return false;

    if(verbose) hpx::cout << "started new work request" << hpx::endl;

    // test if current request is still valid
    dispose_current_request_if_invalid();

    // wait for new request if necessary
    while(!current_request)
    {

        if(verbose) hpx::cout << "no new work. waiting ..." << hpx::endl;
        
        new_request_available.wait(current_request_lock); 
        if(shutdown_requested) return false;
        
        dispose_current_request_if_invalid();
        if(verbose) hpx::cout << "new work! trying again ..." << hpx::endl;
    }

    if(verbose) hpx::cout << "new work aquired. calculating new workload ..." << hpx::endl;

    // calculate current coords
    double workpacket_pos_x = current_topleft_x 
                              + current_vert_pixdist_x * current_img_pos * 
                              current_request->lines_per_gpu;
    double workpacket_pos_y = current_topleft_y 
                              + current_vert_pixdist_y * current_img_pos * 
                              current_request->lines_per_gpu;

    // TODO calculate new workload
    *new_work = boost::make_shared<workload>(
                            current_request->tilesize_x,
                            current_request->lines_per_gpu,
                            workpacket_pos_x,
                            workpacket_pos_y,
                            current_hor_pixdist_x,
                            current_hor_pixdist_y,
                            current_vert_pixdist_x,
                            current_vert_pixdist_y,
                            current_request_id,
                            0,
                            current_img_pos*current_request->lines_per_gpu,
                            current_request->tilesize_x);
    
    // set the next position in image
    current_img_pos++;

    // delete workload if we are the last bit
    if(current_img_pos * current_request->lines_per_gpu >=
                                                    current_request->tilesize_y)
    {
        current_request = boost::shared_ptr<request>();

        // fetch new image
        start_getting_new_image();
    }

    return true;

}

void
maps_image_generator::
start_getting_new_image()
{

    hpx::async(&maps_image_generator::get_new_image, this).then(
        hpx::util::bind(

            // if no new image can be fetched, shut down generator
            [] (
                maps_image_generator* img_gen,
                hpx::lcos::future<bool> parent_future
            ) {

                if(!parent_future.get())
                    img_gen->shutdown();

            },

            this,
            hpx::util::placeholders::_1

        ));

}

void
maps_image_generator::
worker_deliver(boost::shared_ptr<workload>& done_work)
{

    
    if(verbose) hpx::cout << "got delivery from worker." << hpx::endl;
    done_work_queue.push(done_work);
    if(verbose) hpx::cout << "finished delivery from worker." << hpx::endl;

}

void
maps_image_generator::
add_worker(hpx::opencl::device & device, size_t num_parallel_kernels)
{

        // create request callback function for worker
        boost::function<bool(boost::shared_ptr<workload>*)> request_new_work = 
               boost::bind(&maps_image_generator::worker_request_new_work,
                           this,
                           _1); 

        // create deliver callback function for worker
        boost::function<void(boost::shared_ptr<workload>&)> deliver_done_work = 
               boost::bind(&maps_image_generator::worker_deliver,
                           this,
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
        workers.push_back(worker);

}

bool
maps_image_generator::
get_new_image()
{

    // lock the access tu the current request
    boost::lock_guard<hpx::lcos::local::spinlock>
    lock(current_request_lock);
    
    // don't do anything if there still is a current request
    if(current_request)
        return true;

    // get new request
    boost::shared_ptr<request> new_request = acquire_new_request();

    // shutdown if acquire_new_request returned an invalid value
    if(!new_request) return false;
    
    // if image is already dead, abort image and query another one
    if(!new_request->stillValid())
    {
        new_request->abort();
        start_getting_new_image();
        return true;
    }

    // allocate image data
    new_request->data = boost::make_shared<std::vector<char>>(
                                                new_request->tilesize_x 
                                                * new_request->tilesize_y
                                                * 3 * sizeof(char));

    // get new image id
    size_t new_image_id = next_image_id++;
     
    {                                           
        // lock the data list
        boost::lock_guard<hpx::lcos::local::spinlock>
        lock2(images_lock);
    
        // insert new request to images list
        images.insert(std::pair<size_t, boost::shared_ptr<request>>(new_image_id,
                                                                    new_request));
    }

    // set as current request
    current_request = new_request;

    // set current request id
    current_request_id = new_image_id;

    // set current image position
    current_img_pos = 0;
    
    ///////////////////////////////////////////////
    // map raw values to double values 

    // calculate actual zoom
    double zoom = exp2((double)new_request->zoom);

    // calculate sidelength
    double sqrt_2 = sqrt(2.0);
    double tilesidelength = (4.0/sqrt_2) / zoom;

    // calculate actual positions
    double bound = exp2(new_request->zoom);
    double posx = (new_request->posx - bound/2.0 + 0.5) * tilesidelength;
    double posy = -(new_request->posy - bound/2.0 + 0.5) * tilesidelength;

    ///////////////////////////////////////////
    // calculate image coords

    size_t img_width = new_request->tilesize_x;
    size_t img_height = new_request->tilesize_y;

    // calculate aspect ratio
    double aspect_ratio = (double) img_width 
                            / (double) img_height;

    // calculate size of diagonale
    //double size_diag = exp2(-zoom) * 4.0;
    double size_diag = 4.0 / zoom;
    
    // calculate width and height
    double size_y = size_diag / sqrt( 1 + aspect_ratio * aspect_ratio );
    double size_x = aspect_ratio * size_y;

    // calculate horizontal stepwidth
    double rotation = 0.0;
    double hor_pixdist_nonrot = size_x / img_width;
    current_hor_pixdist_x = cos(rotation) * hor_pixdist_nonrot;
    current_hor_pixdist_y = sin(rotation) * hor_pixdist_nonrot;

    // calculate vertical stepwidth
    double vert_pixdist_nonrot = - size_y / img_height;
    current_vert_pixdist_x = - sin(rotation) * vert_pixdist_nonrot;
    current_vert_pixdist_y = cos(rotation) * vert_pixdist_nonrot;


    // calculate top left coords
    current_topleft_x = posx - current_hor_pixdist_x * ( img_width / 2.0 - 0.5 ) 
                          - current_vert_pixdist_x * ( img_height / 2.0 - 0.5 );
    current_topleft_y = posy - current_hor_pixdist_y * ( img_width / 2.0 - 0.5 ) 
                          - current_vert_pixdist_y * ( img_height / 2.0 - 0.5 );


    // signal waiting threads
    new_request_available.notify_all();
    
    hpx::cout << "Started working on " << current_request->zoom
              << " - (" << posx << ", " << posy << ")" << hpx::endl;


    return true;

}

void
maps_image_generator::
wait_for_startup_finished()
{

    // wait for all workers to finish startup
    for( auto& worker : workers)
    {

        worker->wait_for_startup_finished();

    }


}

void 
maps_image_generator::
shutdown()
{

    // set shutdown requested flag
    shutdown_requested = true;
         
    // signal workers to continue working, they will then read the
    // shutdown_requested flag and end
    new_request_available.notify_all(); 

    // wait for all workers to finish
    for( auto& worker : workers)
    {
        worker->join(); 
    }

    // then, signal the retrievers to shutdown
    done_work_queue.finish();    

    // wait for retrievers to finish
    retrievers_finished.wait();

}

void
maps_image_generator::
retrieve_worker_main(intptr_t parent_, bool verbose)
{

    // get parent pointer
    maps_image_generator* parent = (maps_image_generator*) parent_;

    // represents done workload
    boost::shared_ptr<workload> done_workload;

    // main loop
    if(verbose) hpx::cout << "entering retrieve worker main loop ..." << hpx::endl;
    while(parent->done_work_queue.pop(&done_workload))
    {

        if(verbose) hpx::cout << "retrieved workload "
                              << done_workload->pos_in_img_x
                              << ":" 
                              << done_workload->pos_in_img_y
                              << hpx::endl;

        // retrieve id of associated image
        size_t img_id = done_workload->img_id;

        // image data
        boost::shared_ptr<request> img_request;

        // retrieve image pointers
        {
            // lock
            boost::lock_guard<hpx::lcos::local::spinlock>
            lock(parent->images_lock);
            
            // try to find the associated request
            image_request_map::iterator req_iterator = parent->images.find(img_id);
            // indicates that the image is gone. don't handle data in this case.
            if(req_iterator == parent->images.end())
                continue;

            // read the request
            img_request = req_iterator->second;
        }

        // copy data to img_data
        size_t start_x = done_workload->pos_in_img_x;
        size_t start_y = done_workload->pos_in_img_y;
        size_t size_x = done_workload->num_pixels_x;
        size_t size_y = done_workload->num_pixels_y;
        size_t line_offset = done_workload->line_offset;
        for(size_t y = 0; y < size_y; y++)
        {
            for(size_t x = 0; x < size_x; x++)
            {
                (*(img_request->data))
                       [((y + start_y) * line_offset + (x + start_x)) * 3 + 0] =
                    done_workload->pixeldata[(y * size_x + x) * 3 + 0];  
                (*(img_request->data))
                       [((y + start_y) * line_offset + (x + start_x)) * 3 + 1] =
                    done_workload->pixeldata[(y * size_x + x) * 3 + 1];  
                (*(img_request->data))
                       [((y + start_y) * line_offset + (x + start_x)) * 3 + 2] =
                    done_workload->pixeldata[(y * size_x + x) * 3 + 2];  
            }
        }

        // decrease the number of work packets left
        size_t current_img_countdown = --(img_request->img_countdown);

        // if no work packet left (img finished), then:
        if(current_img_countdown == 0)
        {

            // convert to png
            if(img_request->stillValid())
            {
                size_t png_size;
                boost::shared_array<char> png_data = 
                                   create_png(img_request->data,
                                              img_request->tilesize_x,
                                              img_request->tilesize_y,
                                              &png_size);
                
                // remove old data
                img_request->data = boost::make_shared<std::vector<char>>
                                                (png_data.get(),
                                                 png_data.get() + png_size);
    
                // send data
                img_request->done(img_request->data);
            } else {
                img_request->abort();   
            }

            // lock the data lists
            boost::lock_guard<hpx::lcos::local::spinlock>
            lock(parent->images_lock);

            // remove image data.
            // data will still be available for waiting image thread,
            // as it is a shared_ptr.
            image_request_map::iterator img_it = parent->images.find(img_id);
            if(img_it != parent->images.end())
                parent->images.erase(img_it);

        }

    }

}


/*
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
maps_image_generator::
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
maps_image_generator::
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
    boost::shared_ptr<std::atomic_size_t> img_countdown = 
              boost::make_shared<std::atomic_size_t>(num_tiles_x * num_tiles_y);

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
                                boost::shared_ptr<std::atomic_size_t>>
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

}*/
