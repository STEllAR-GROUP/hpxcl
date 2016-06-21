// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPXCL_MANDELBROT_REQUESTHANDLER_HPP_
#define HPXCL_MANDELBROT_REQUESTHANDLER_HPP_

#include <hpx/hpx.hpp>
#include <string>

#include "maps_image_generator.hpp"
#include "../fifo.hpp"
#include <atomic>
#include <memory>

namespace hpx { namespace opencl { namespace examples { namespace mandelbrot {

struct request
{
public:
    boost::function<bool(void)> stillValid;
    boost::function<void(std::shared_ptr<std::vector<char>>)> done;
    boost::function<void(void)> abort;
    long zoom;
    long posx;
    long posy;
    std::string user_ip;
    std::shared_ptr<std::vector<char>> data;
    size_t tilesize_x;
    size_t tilesize_y;
    size_t lines_per_gpu;
    std::atomic<size_t> img_countdown;
};


class requesthandler
{

public:
    // constructor
    requesthandler(size_t tilesize_x_,
                   size_t tilesize_y_,
                   size_t lines_per_gpu);

    void submit_request(std::shared_ptr<request> request);

    std::shared_ptr<request> query_request();


private:
    size_t tilesize_x;
    size_t tilesize_y;
    size_t lines_per_gpu;
    fifo<std::shared_ptr<request>> new_requests;


};



} } } }

#endif
