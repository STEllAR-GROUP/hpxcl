// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "requesthandler.hpp"

#include <hpx/include/iostreams.hpp>

using namespace hpx::opencl::examples::mandelbrot;

requesthandler::requesthandler(size_t tilesize_x_,
                               size_t tilesize_y_,
                               size_t lines_per_gpu_) :
                                    tilesize_x(tilesize_x_),
                                    tilesize_y(tilesize_y_),
                                    lines_per_gpu(lines_per_gpu_)
{
    
    
}

void
requesthandler::submit_request(boost::shared_ptr<request> request)
{
 
    // Check if still valid
    if(!request->stillValid())
    {
        request->abort();
        return;
    }

    // add missing data in request
    request->tilesize_x = tilesize_x;
    request->tilesize_y = tilesize_y;
    request->lines_per_gpu = lines_per_gpu;
    request->img_countdown = tilesize_y/lines_per_gpu;
      
    // hand the request to an hpx thread    
    new_requests.push(request);
    
}

boost::shared_ptr<request>
requesthandler::query_request()
{

    boost::shared_ptr<request> ret;

    // take a new request out of the queue
    while(true)
    {
        if(!new_requests.pop(&ret))
            return boost::shared_ptr<request>(); 
        if(ret->stillValid())
        {
            return ret;
        }
        ret->abort();
    }

}


