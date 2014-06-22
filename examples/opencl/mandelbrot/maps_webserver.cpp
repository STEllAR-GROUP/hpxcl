// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/include/iostreams.hpp>

#include "maps_webserver.hpp"

#include "../../../opencl.hpp"

#include "mandelbrot_ico_data.hpp"
#include "mandelbrot_html_data.hpp"

#include "pngwriter.hpp"
#include "image_generator.hpp"
#include "mongoose.hpp"

#include <string>
#include <cmath>

static image_generator* img_gen;
static size_t tilesize_x;
static size_t tilesize_y;
static size_t lines_per_gpu;

static bool handle_special_file(struct mg_connection *conn)
{

    std::string uri(conn->uri);

    if(uri.size() < 1) return false;

    // Send main file
    if(uri.compare("/") == 0)
    {
        hpx::cout << "Sending main html file." << hpx::endl;
 
        mg_printf(conn,
                  "HTTP/1.1 200 OK\r\n"
                  "Content-Type: text/html; charset=utf-8\r\n"
                  "Content-Length: %zu\r\n"
                  "Connection: close\r\n\r\n",
                  mandelbrot_html_len);

        mg_write(conn, mandelbrot_html, mandelbrot_html_len);
            
        mg_write(conn, "\r\n", 2);
       
        return true;
    }

    // Send icon
    if(uri.compare("/favicon.ico") == 0)
    {
        hpx::cout << "Sending website icon." << hpx::endl;

        mg_printf(conn,
                  "HTTP/1.1 200 OK\r\n"
                  "Content-Type: image/x-icon\r\n"
                  "Content-Length: %zu\r\n"
                  "Connection: close\r\n\r\n",
                  mandelbrot_ico_len);

        mg_write(conn, mandelbrot_ico, mandelbrot_ico_len);
            
        mg_write(conn, "\r\n", 2);

        return true;
    }

    return false;
}

static boost::shared_ptr<std::vector<char>> generate_image(std::string uri)
{

    boost::shared_ptr<std::vector<char>> ret;

    // remove '/'
    if(uri.size() < 1) return ret;
    if(uri[0] != '/') return ret;
    uri = uri.substr(1);

    size_t pos;

    // split first number
    pos = uri.find('/');
    if(pos == uri.npos) return ret;
    std::string zoom_string = uri.substr(0, pos);
    uri = uri.substr(pos+1);

    // split second number
    pos = uri.find('/');
    if(pos == uri.npos) return ret;
    std::string pos_x_string = uri.substr(0, pos);
    uri = uri.substr(pos+1);

    // split third number
    pos = uri.find('.');
    if(pos == uri.npos) return ret;
    std::string pos_y_string = uri.substr(0, pos);
    uri = uri.substr(pos+1);

    // check for correct file format requested
    if(uri.compare("png") != 0) return ret;

    // convert strings to long
    long zoom_raw;
    long pos_x_raw;
    long pos_y_raw;
    try {
        zoom_raw = std::stol(zoom_string); 
        pos_x_raw = std::stol(pos_x_string); 
        pos_y_raw = std::stol(pos_y_string); 
    } catch ( std::exception e ) {
        return ret;
    }

    // move image to the middle

    // calculate actual zoom
    double zoom = exp2((double)zoom_raw);

    // calculate sidelength
    double sqrt_2 = sqrt(2.0);
    double tilesidelength = (4.0/sqrt_2) / zoom;

    // calculate actual positions
    double bound = exp2(zoom_raw);
    double pos_x = (pos_x_raw - bound/2.0 + 0.5) * tilesidelength;
    double pos_y = -(pos_y_raw - bound/2.0 + 0.5) * tilesidelength;

    hpx::cout << "Requested tile: " << zoom << " - (" << pos_x << "," << pos_y << ")" << hpx::endl;
    //std::cout << "Requested tile: " << zoom << " - (" << pos_x << "," << pos_y << ")" << std::endl;

    // calculate image 
    return      img_gen->compute_image(pos_x,
                                       pos_y,
                                       zoom,
                                       0.0,
                                       tilesize_x,
                                       tilesize_y,
                                       false,
                                       tilesize_x,
                                       lines_per_gpu).get();



}

static int event_handler(struct mg_connection *conn, enum mg_event ev) {

  switch (ev) {
    case MG_AUTH: return MG_TRUE;
    case MG_REQUEST:
        {
            
            hpx::cout << "REQUESTED: '" << conn->uri << "'" << hpx::endl;

            if(handle_special_file(conn))
                return MG_TRUE;

            // Calculate image
            boost::shared_ptr<std::vector<char>> img_data = generate_image(conn->uri);
            if(!img_data) return MG_FALSE;

            // Compress image
            size_t png_size;
            boost::shared_array<char> png =
                create_png( img_data,
                            tilesize_x,
                            tilesize_y,
                            &png_size );

            mg_printf(conn,
                      "HTTP/1.1 200 OK\r\n"
                      "Content-Type: image/png\r\n"
                      "Content-Length: %zu\r\n"
                      "Connection: close\r\n\r\n",
                      png_size);

            mg_write(conn, png.get(), png_size);
            
            mg_write(conn, "\r\n", 2);
        }
        return MG_TRUE;
        
    default: return MG_FALSE;
  }
}

static void serve(void *server)
{
    for (;;) mg_poll_server((struct mg_server *) server, 1000);
}

void run_webserver(const char* port,
                   image_generator *img_gen_,
                   size_t tilesize_x_,
                   size_t tilesize_y_,
                   size_t lines_per_gpu_,
                   size_t num_threads)
{

    // Set the generator
    img_gen = img_gen_;
    tilesize_x = tilesize_x_;
    tilesize_y = tilesize_y_;
    lines_per_gpu = lines_per_gpu_;

    // Create the server
    std::vector<mg_server*> servers;
    std::vector<hpx::future<void>> server_futures;
    for(size_t i = 0; i < num_threads; i++)
    {
        mg_server* server = mg_create_server(NULL, event_handler);

        if(servers.empty())
        {
            mg_set_option(server, "listening_port", "8080");
        }
        else
        {
            mg_set_listening_socket(server, mg_get_listening_socket(servers[0]));
        }

        servers.push_back(server);
    }

    // Start servers
    for(size_t i = 0; i < num_threads; i++)
    {
        server_futures.push_back(hpx::async(serve, servers[i]));
    }

    for(size_t i = 0; i < num_threads; i++)
    {
        server_futures[i].wait();
        mg_server* server = servers[i];
        mg_destroy_server(&server);
    }

    // Should not get reached
    
}

