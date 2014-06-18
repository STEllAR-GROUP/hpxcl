// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "pngwriter.hpp"

#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL

#include <boost/shared_ptr.hpp>
#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/png_dynamic_io.hpp>

#include <vector>

using namespace boost::gil;


void save_png(boost::shared_ptr<std::vector<char>> data, size_t width, size_t height, const char* filename)
{
    
    // create an image
    rgb8_image_t img(width, height);

    // create a view to the image
    rgb8_image_t::view_t v = view(img);

    // iterate through all rows
    for(size_t y = 0; y < height; y++)
    {
        // create a row iterator
        rgb8_image_t::view_t::x_iterator it = v.row_begin(y);
            
        // set data of the row
        for(size_t x = 0; x < width; x++)
        {
            *it = rgb8_pixel_t((unsigned char)((*data)[(y * width + x) * 3 + 0]), 
                               (unsigned char)((*data)[(y * width + x) * 3 + 1]),
                               (unsigned char)((*data)[(y * width + x) * 3 + 2]));
            it++;
        }
    }
 
    // write to file
    boost::gil::png_write_view(filename, const_view(img));

}
