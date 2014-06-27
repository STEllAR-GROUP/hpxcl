// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "requesthandler.hpp"

#include <hpx/include/iostreams.hpp>

using namespace hpx::opencl::examples::mandelbrot;

requesthandler::requesthandler(image_generator * img_gen_,
               size_t tilesize_x_,
               size_t tilesize_y_,
               size_t lines_per_gpu)
{
    
    
}

void
requesthandler::submit_request(boost::shared_ptr<request> request)
{
 
    std::cout << "Request from " << request->user_ip
              << ": " << request->zoom 
              << " - (" << request->posx
              << "," << request->posy
              << ")" << std::endl; 

    boost::shared_ptr<std::vector<char>> res = 
                            boost::make_shared<std::vector<char>>();

    res->push_back('A');
    res->push_back('B');
    res->push_back('C');
    res->push_back('\n');


    request->done(res);

//    request->abort();
    
}

 
