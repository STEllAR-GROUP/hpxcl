// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MANDELBROT_PNG_WRITER_H_
#define MANDELBROT_PNG_WRITER_H_

#include <cstdlib>

// creates an image with size x,y
unsigned long png_create(size_t x, size_t y);
void test();

// sets a row of the image. data needs to be an array of size 3*x
void png_set_row(unsigned long id, size_t y, unsigned char * data);
void png_save_and_close(unsigned long id, const char* filename);

#endif

