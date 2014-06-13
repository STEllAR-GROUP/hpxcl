// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MANDELBROT_WORKLOAD_H_
#define MANDELBROT_WORKLOAD_H_

#include <cstdlib>
#include <vector>

#include <boost/shared_ptr.hpp>

/*
 * A workload, defines a mandelbrot line and will be filled by workers with
 * the calculated results
 */
class workload
{

    public:
        workload(size_t num_pixels_, double origin_x_, double origin_y_,
                 double size_x_, double size_y_, size_t img_id_,
                 size_t pos_in_img_);
        
        // Will hold the calculated pixels
        boost::shared_ptr<std::vector<char>> pixeldata;
        // the number of pixels on the line
        size_t num_pixels;
        // the start point of the line
        double origin_x;
        double origin_y;
        // the end point of the line
        double size_x;
        double size_y;
        // metadata for correct mapping to image
        size_t img_id;
        size_t pos_in_img;

};

#endif

