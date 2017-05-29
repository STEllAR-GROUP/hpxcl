// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPXCL_MANDELBROT_WEBSERVER_HPP_
#define HPXCL_MANDELBROT_WEBSERVER_HPP_

#include <hpx/hpx.hpp>
#include <hpx/config/asio.hpp>
#include <hpx/include/runtime.hpp>
#include "requesthandler.hpp"

#include <boost/asio.hpp>
#include <hpx/compat/thread.hpp>

#include <memory>

namespace hpx { namespace opencl { namespace examples { namespace mandelbrot {

class webserver
{

public:

    // constructor
    webserver(unsigned short port, requesthandler * req_handler_);

    // starts the webserver
    void start();

    // stops the webserver
    void stop();


private:
    // the main function of the webserver
    void external_start(hpx::runtime* rt);

    // enqueues listening for one new connection
    void start_listening_for_new_connection();

    // gets called when a new connection got established
    void new_connection_callback(
                        const boost::system::error_code & error,
                        std::shared_ptr<boost::asio::ip::tcp::socket> socket);

    // starts reading a new line from a socket, async call
    void start_reading_line_from_socket(
                        std::shared_ptr<boost::asio::streambuf> buffer,
                        std::shared_ptr<boost::asio::ip::tcp::socket> socket,
                        size_t lines_read,
                        std::string requested_filename);

    // gets called when a new line got read from a socket
    void new_line_read_callback(
                        const boost::system::error_code & error,
                        size_t bytes_transferred,
                        std::shared_ptr<boost::asio::ip::tcp::socket> socket,
                        std::shared_ptr<boost::asio::streambuf> buffer,
                        size_t lines_read,
                        std::string requested_filename);

    // callback, closes the socket and returns.
    // keeps data alive until the write finishes
    void close_socket(std::shared_ptr<boost::asio::ip::tcp::socket> socket,
                      boost::any keepalive_data);

    // keeps data alive until the write finishes
    void dont_close_socket(boost::any keepalive_data);

    // sends '500 Server Error' and closes the socket
    void send_server_error_and_close(
                        std::shared_ptr<boost::asio::ip::tcp::socket> socket);

    // sends '400 Bad Request' and closes the socket
    void send_bad_request_and_close(
                        std::shared_ptr<boost::asio::ip::tcp::socket> socket);

    // sends '404 Not Found' and closes the socket
    void send_not_found_and_close(
                        std::shared_ptr<boost::asio::ip::tcp::socket> socket);

    // reads the first http request line.
    // returns the queried filename or "" if an error occured
    std::string read_filename_from_request(std::string line);

    // gets called once per request.
    // this function reads the filename and creates a corresponding answer.
    void process_request(std::shared_ptr<boost::asio::ip::tcp::socket> socket,
                         std::string filename);

    // sends data to the client and sends '100 Continue'
    void send_data(std::shared_ptr<boost::asio::ip::tcp::socket> socket,
                   const char* content_type,
                   const char* data,
                   size_t data_size);

    // sends data to the client and sends '100 Continue'
    void send_data(
                         std::shared_ptr<boost::asio::ip::tcp::socket> socket,
                         const char* content_type,
                         std::shared_ptr<std::vector<char>> data);

    // checks wether a socket is connected or not
    bool is_socket_still_connected(
                        std::shared_ptr<boost::asio::ip::tcp::socket> socket);


private:
    requesthandler * req_handler;
    boost::asio::io_service io_service;
    boost::asio::ip::tcp::acceptor acceptor;
    boost::asio::strand strand;

    // the main external worker thread
    hpx::compat::thread asio_thread;

};



} } } }

#endif


