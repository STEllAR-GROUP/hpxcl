// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MANDELBROT_PNG_WRITER_H_
#define MANDELBROT_PNG_WRITER_H_


#include <cstdlib>
#include <vector>
#include <memory>
#include <boost/shared_array.hpp>

// writes data to png file
void save_png(std::shared_ptr< std::vector<char> > data, size_t width, size_t height, const char* filename);

boost::shared_array<char> create_png(std::shared_ptr< std::vector<char> > data, size_t width, size_t height, size_t * size);

void png_write_to_file(boost::shared_array<char> png, size_t png_size, const char* filename);

void save_pngi_it(std::shared_ptr< std::vector<char> > data, size_t width, size_t height, size_t it);
#endif

