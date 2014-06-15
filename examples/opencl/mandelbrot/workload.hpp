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
        workload(size_t num_pixels_x_,
                 size_t num_pixels_y_,
                 double topleft_x_,
                 double topleft_y_,
                 double topright_x_,
                 double topright_y_,
                 double botleft_x_,
                 double botleft_y_,
                 size_t img_id_,
                 size_t pos_in_img_x_,
                 size_t pos_in_img_y_,
                 size_t line_offset_);
        
        // Will hold the calculated pixels
        boost::shared_ptr<std::vector<char>> pixeldata;
        // the number of pixels on the rectangle
        size_t num_pixels_x;
        size_t num_pixels_y;
        // the top left point of the rectangle
        double topleft_x;
        double topleft_y;
        // the top right point of the rectangle
        double topright_x;
        double topright_y;
        // the bottom left point of the rectangle
        double botleft_x;
        double botleft_y;
        // metadata for correct mapping to image
        size_t img_id;
        size_t pos_in_img_x;
        size_t pos_in_img_y;
        size_t line_offset;

};

#endif

