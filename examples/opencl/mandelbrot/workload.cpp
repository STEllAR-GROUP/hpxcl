// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "workload.hpp"

workload::workload(size_t num_pixels_, double origin_x_, double origin_y_,
                    double size_x_, double size_y_, size_t img_id_,
                    size_t pos_in_img_) :
                    pixeldata(std::vector<unsigned char>(3*num_pixels_)),
                    num_pixels(num_pixels_),
                    origin_x(origin_x_),
                    origin_y(origin_y_),
                    size_x(size_x_),
                    size_y(size_y_),
                    img_id(img_id_),
                    pos_in_img(pos_in_img_){}
        

