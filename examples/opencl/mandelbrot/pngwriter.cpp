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

#include <atomic>
#include <map>

using namespace boost::gil;




// an image, basically a struct that prevents repeated view creation
class png_img{

public:
    // constructor
    png_img(){}
    png_img(size_t x_, size_t y_): img(x_, y_), v(view(img))
    {
    }

    void set_row(size_t y, unsigned char* data)
    {
        
        // create a row iterator
        rgb8_image_t::view_t::x_iterator it = v.row_begin(y);
        
        // set data of the row
        for(int x = 0; x < img.width(); x++)
        {
            *it = rgb8_pixel_t(data[0], data[1], data[2]);
            data += 3;
            it++;
        }

    }

    void save_to_file(const char* filename){
        
        boost::gil::png_write_view(filename, const_view(img));

    }

private:
    // saves the image itself
    rgb8_image_t img;
    // view, used to modify the image
    rgb8_image_t::view_t v;

};


static std::atomic_ulong next_id(0);
static std::map<unsigned long, boost::shared_ptr<png_img>> images;

unsigned long png_create(size_t x, size_t y) 
{
    // query new image id
    int id = next_id++;

    // create image
    boost::shared_ptr<png_img> img_ptr(new png_img(x,y));

    // store image
    images.insert(
             std::pair<unsigned long, boost::shared_ptr<png_img>>(id, img_ptr));

    // return reference id
    return id;

}

void png_set_row(unsigned long id, size_t y, unsigned char* data)
{

    images[id]->set_row(y, data);

}

void png_save_and_close(unsigned long id, const char* filename)
{

    // write image to file
    images[id]->save_to_file(filename);

    // delete the image
    images.erase(id);

}


