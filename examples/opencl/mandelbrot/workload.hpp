// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MANDELBROT_WORKLOAD_H_
#define MANDELBROT_WORKLOAD_H_

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>

#include <cstdlib>
#include <memory>
#include <vector>

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
                 double hor_pixdist_x_,
                 double hor_pixdist_y_,
                 double vert_pixdist_x_,
                 double vert_pixdist_y_,
                 size_t img_id_,
                 size_t pos_in_img_x_,
                 size_t pos_in_img_y_,
                 size_t line_offset_);

        // Will hold the calculated pixels
        hpx::serialization::serialize_buffer<char> pixeldata;
        // the number of pixels on the rectangle
        size_t num_pixels_x;
        size_t num_pixels_y;
        // the top left point of the rectangle
        double topleft_x;
        double topleft_y;
        // the horizontal offset between pixels
        double hor_pixdist_x;
        double hor_pixdist_y;
        // the vertical offset between pixels
        double vert_pixdist_x;
        double vert_pixdist_y;
        // metadata for correct mapping to image
        size_t img_id;
        size_t pos_in_img_x;
        size_t pos_in_img_y;
        size_t line_offset;

};

#endif

