// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MAPS_WEBSERVER_HPP_
#define MAPS_WEBSERVER_HPP_

#include "image_generator.hpp"


void run_webserver(const char* port,
                   image_generator *imggen_,
                   size_t tilesize_x,
                   size_t tilesize_y,
                   size_t lines_per_gpu,
                   size_t num_threads);





#endif
