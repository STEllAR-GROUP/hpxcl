// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPXCL_MANDELBROT_REQUESTHANDLER_HPP_
#define HPXCL_MANDELBROT_REQUESTHANDLER_HPP_

#include <hpx/hpx.hpp>
#include <string>

#include "../image_generator.hpp"

#include <boost/shared_ptr.hpp>


namespace hpx { namespace opencl { namespace examples { namespace mandelbrot {

struct request
{
public:
    boost::function<bool(void)> stillValid;
    boost::function<void(boost::shared_ptr<std::vector<char>>)> done;
    boost::function<void(void)> abort;
    long zoom;
    long posx;
    long posy;
    std::string user_ip;
};


class requesthandler
{

public:
    // constructor
    requesthandler(image_generator * img_gen_,
                   size_t tilesize_x_,
                   size_t tilesize_y_,
                   size_t lines_per_gpu);

    void submit_request(boost::shared_ptr<request> request);

     




};



} } } }

#endif
