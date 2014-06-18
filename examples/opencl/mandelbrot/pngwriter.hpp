// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MANDELBROT_PNG_WRITER_H_
#define MANDELBROT_PNG_WRITER_H_

#include <cstdlib>
#include <boost/shared_ptr.hpp>

// writes data to png file
void save_png(boost::shared_ptr<std::vector<char>> data, size_t width, size_t height, const char* filename);

#endif

