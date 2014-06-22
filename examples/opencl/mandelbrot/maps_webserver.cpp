// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/include/iostreams.hpp>

#include "maps_webserver.hpp"

#include "../../../opencl.hpp"


#include "pngwriter.hpp"
#include "image_generator.hpp"
#include "mongoose.hpp"


static image_generator* img_gen;
static size_t tilesize_x;
static size_t tilesize_y;
static size_t lines_per_gpu;

static int event_handler(struct mg_connection *conn, enum mg_event ev) {

  switch (ev) {
    case MG_AUTH: return MG_TRUE;
    case MG_REQUEST:
        {
            hpx::cout << "Requested: " << conn->uri << hpx::endl;
            
            // Calculate image
            boost::shared_ptr<std::vector<char>> img_data =
                img_gen->compute_image(-0.7,
                                       0.0,
                                       1.04,
                                       0.0,
                                       tilesize_x,
                                       tilesize_y,
                                       false,
                                       tilesize_x,
                                       lines_per_gpu).get();

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

