// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#include "pngwriter.hpp"

#include <fstream>
#include <png.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

/* structure to store PNG image bytes */
struct mem_encode {
  char *buffer;
  size_t size;
};

static void my_png_write_data(png_structp png_ptr, png_bytep data,
                              png_size_t length) {
  /* with libpng15 next line causes pointer deference error; use libpng12 */
  struct mem_encode *p =
      (struct mem_encode *)png_get_io_ptr(png_ptr); /* was png_ptr->io_ptr */
  size_t nsize = p->size + length;

  /* allocate or grow buffer */
  if (p->buffer)
    p->buffer = (char *)realloc(p->buffer, nsize);
  else
    p->buffer = (char *)malloc(nsize);

  if (!p->buffer) png_error(png_ptr, "Write Error");

  /* copy new bytes to end of buffer */
  memcpy(p->buffer + p->size, data, length);
  p->size += length;
}

#define die(func, msg) \
  { HPX_THROW_EXCEPTION(hpx::no_success, (func), (msg)); }

static mem_encode save_png_to_mem(std::shared_ptr<std::vector<char> > data,
                                  size_t width, size_t height) {
  png_structp png_ptr = NULL;
  png_infop info_ptr = NULL;
  size_t y;
  png_uint_32 bytes_per_row;
  png_byte **row_pointers = NULL;

  // set up png_ptr
  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (png_ptr == NULL) {
    die("png_create_write_struct()", "Returned NULL");
  }

  // set up info_ptr
  info_ptr = png_create_info_struct(png_ptr);
  if (info_ptr == NULL) {
    png_destroy_write_struct(&png_ptr, NULL);
    die("png_create_info_stuct()", "Returned NULL");
  }

  /* Set up error handling. */
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_write_struct(&png_ptr, &info_ptr);
    die("save_png_to_mem()", "Error callback called!");
  }

  /* Set image attributes. */
  png_set_IHDR(png_ptr, info_ptr, (png_uint_32)width, (png_uint_32)height, 8,
               PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

  /* Initialize the rows of png */
  bytes_per_row = (png_uint_32)(width * sizeof(char) * 3);
  row_pointers = (png_byte **)png_malloc(png_ptr, height * sizeof(png_byte *));
  for (y = 0; y < height; ++y) {
    row_pointers[y] = (png_byte *)(data->data() + 3 * y * width);
  }

  /* static */
  struct mem_encode state;

  /* initialise - put this before png_write_png() call */
  state.buffer = NULL;
  state.size = 0;

  /* if my_png_flush() is not needed, change the arg to NULL */
  png_set_write_fn(png_ptr, &state, my_png_write_data, NULL);

  /* the actual write */
  png_set_rows(png_ptr, info_ptr, row_pointers);
  png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

  /* Cleanup. */
  png_free(png_ptr, row_pointers);

  /* Finish writing. */
  png_destroy_write_struct(&png_ptr, &info_ptr);

  /* now state.buffer contains the PNG image of size s.size bytes */

  return state;
}

boost::shared_array<char> create_png(std::shared_ptr<std::vector<char> > data,
                                     size_t width, size_t height,
                                     size_t *size) {
  // Create png in memory
  mem_encode png_data = save_png_to_mem(data, width, height);

  // Wrap png in shared_array for auto-deletion
  boost::shared_array<char> png(png_data.buffer);

  // write size to external variable
  *size = png_data.size;

  // return the png
  return png;
}

void png_write_to_file(boost::shared_array<char> png, size_t png_size,
                       const char *filename) {
  // Open file
  std::ofstream file(filename,
                     std::ios::out | std::ios::binary | std::ios::trunc);

  // Ensure that file is open
  if (!file.is_open()) {
    die("png_write_to_file()", "Can't open file!");
  }

  // Write to file
  file.write(png.get(), png_size);

  // Close file
  file.close();
}

void save_png(std::shared_ptr<std::vector<char> > data, size_t width,
              size_t height, const char *filename) {
  size_t png_size;

  boost::shared_array<char> png = create_png(data, width, height, &png_size);

  png_write_to_file(png, png_size, filename);
}

void save_png_it(std::shared_ptr<std::vector<char> > data, size_t width,
                 size_t height, size_t it) {
  size_t png_size;

  boost::shared_array<char> png = create_png(data, width, height, &png_size);

  std::string filename;
  filename.append("Mandelbrot_");
  filename.append(std::to_string(it));
  filename.append(".png");

  png_write_to_file(png, png_size, filename.c_str());
}
