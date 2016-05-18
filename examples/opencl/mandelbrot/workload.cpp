// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "workload.hpp"

workload::workload(size_t num_pixels_x_,
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
                   size_t line_offset_)
                   : pixeldata(hpx::serialization::serialize_buffer<char>()),
                     num_pixels_x(num_pixels_x_),
                     num_pixels_y(num_pixels_y_),
                     topleft_x(topleft_x_),
                     topleft_y(topleft_y_),
                     hor_pixdist_x(hor_pixdist_x_),
                     hor_pixdist_y(hor_pixdist_y_),
                     vert_pixdist_x(vert_pixdist_x_),
                     vert_pixdist_y(vert_pixdist_y_),
                     img_id(img_id_),
                     pos_in_img_x(pos_in_img_x_),
                     pos_in_img_y(pos_in_img_y_),
                     line_offset(line_offset_)
                   {};

 

