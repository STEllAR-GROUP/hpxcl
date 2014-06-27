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
 
    std::cout << "Checking for request still valid ..." << std::endl;

    if(!request->stillValid())
    {

        request->abort();
        return;

    }

    std::cout << "Request from " << request->user_ip
                  << ": " << request->zoom 
                  << " - (" << request->posx
                  << "," << request->posy
                  << ")" << std::endl; 

    boost::shared_ptr<std::vector<char>> res = 
                            boost::make_shared<std::vector<char>>();
/*
    res->push_back('A');
    res->push_back('B');
    res->push_back('C');
    res->push_back('\n');

    request->done(res);

//    request->abort();
*/

    new_requests.push(request);
    
}

boost::shared_ptr<request>
requesthandler::query_request()
{

    boost::shared_ptr<request> ret;

    // take a new request out of the queue
    if(new_requests.pop(&ret))
    {
        ret->tilesize_x = tilesize_x;
        ret->tilesize_y = tilesize_y;
        ret->lines_per_gpu = lines_per_gpu;
        ret->img_countdown = tilesize_y/lines_per_gpu;
        return ret;
    }
    else
        // if queue ended, return an empty request to signal shutdown
        return boost::shared_ptr<request>();

}


