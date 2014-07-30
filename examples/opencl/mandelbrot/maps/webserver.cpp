// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "webserver.hpp"

#include <hpx/include/runtime.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <sstream>

#include "resources/resources.hpp"

using namespace hpx::opencl::examples::mandelbrot;
using boost::asio::ip::tcp;

static size_t num_requests = 0;
static size_t num_answers = 0;
static size_t num_aborted = 0;

webserver::webserver(unsigned short port, requesthandler * req_handler_) :
          req_handler(req_handler_),
          io_service(),
          acceptor(io_service, tcp::endpoint(tcp::v4(), port)),
          strand(io_service)
{

     

}


// This struct is used for automatic registration and unregistration of 
// non-hpx threads
struct registration_wrapper                                                      
{                                                                                
    registration_wrapper(hpx::runtime* rt, char const* name)                     
      : rt_(rt)                                                                  
    {                                                                            
        // Register this thread with HPX, this should be done once for           
        // each external OS-thread intended to invoke HPX functionality.         
        // Calling this function more than once will silently fail (will         
        // return false).                                                        
        rt_->register_thread(name);                                              
    }                                                                            
    ~registration_wrapper()                                                      
    {                                                                            
        // Unregister the thread from HPX, this should be done once in the       
        // end before the external thread exists.                                
        rt_->unregister_thread();                                                
    }                                                                            
                                                                                 
    hpx::runtime* rt_;                                                           
}; 

void
webserver::dont_close_socket(boost::any keepalive_data)
{
    std::cout << "Packets: " << num_answers << "/" << num_aborted
              << "/" << num_requests << " - "
              << (num_requests - num_answers - num_aborted) << " lost"
              << std::endl;
}

void
webserver::close_socket(boost::shared_ptr<tcp::socket> socket,
                        boost::any keepalive_data)
{

    socket->close();

}

void
webserver::send_server_error_and_close(boost::shared_ptr<tcp::socket> socket)
{

    //std::cout << "aborted" << std::endl;
    num_aborted ++;

    boost::shared_ptr<std::string> response = boost::make_shared<std::string>(
        "HTTP/1.0 500 Server Error\r\n"
        "Connection: Close\r\n"
        "\r\n");

    boost::asio::async_write(*socket,
                             boost::asio::buffer(*response),
                             strand.wrap(boost::bind(&webserver::close_socket,
                                                     this,
                                                     socket,
                                                     response)));

}

void
webserver::send_not_found_and_close(boost::shared_ptr<tcp::socket> socket)
{

    boost::shared_ptr<std::string> response = boost::make_shared<std::string>(
        "HTTP/1.0 404 Not Found\r\n"
        "Connection: Close\r\n"
        "\r\n");

    boost::asio::async_write(*socket,
                             boost::asio::buffer(*response),
                             strand.wrap(boost::bind(&webserver::close_socket,
                                                     this,
                                                     socket,
                                                     response)));

}

void
webserver::send_bad_request_and_close(boost::shared_ptr<tcp::socket> socket)
{

    boost::shared_ptr<std::string> response = boost::make_shared<std::string>(
        "HTTP/1.0 400 Bad Request\r\n"
        "Connection: Close\r\n"
        "\r\n");

    boost::asio::async_write(*socket,
                             boost::asio::buffer(*response),
                             strand.wrap(boost::bind(&webserver::close_socket,
                                                     this,
                                                     socket,
                                                     response)));

}

std::string
webserver::read_filename_from_request(std::string line)
{

        // vars
        const std::string string_GET("GET ");
        const std::string string_HTTP("HTTP/");
        size_t pos; 

        // if request doesn't start with "GET ", send error
        if(line.compare(0, string_GET.length(), string_GET) != 0)
        {
            // send error
            return "";
        } 

        // cut away "GET "
        line = line.substr(string_GET.length()); 

        // find the next space
        pos = line.find(' ');
        if(pos == line.npos)
        {
            // send error
            return "";
        }

        // take request filename
        std::string filename = line.substr(0, pos);

        // cut away filename
        line = line.substr(pos + 1);

        // ensure it's a http request 
        if(line.compare(0, string_HTTP.length(), string_HTTP) != 0)
        {
            // send error
            return "";
        }

        return filename;
}

struct send_data_data
{
    std::string header = "";
    std::string footer = "";
    boost::shared_ptr<std::vector<char>> data;
    std::vector<boost::asio::const_buffer> buffers;
};

void
webserver::send_data(boost::shared_ptr<tcp::socket> socket,
                     const char* content_type,
                     boost::shared_ptr<std::vector<char>> data)
{
    
    num_answers ++;

    // store all the data that we need to keep alive
    boost::shared_ptr<send_data_data> keep_alive_data = boost::make_shared<send_data_data>();
    
    // keep data ptr alive
    keep_alive_data->data = data;

    // vector to keep data alive
    boost::shared_ptr<std::vector<boost::any>> keep_= boost::make_shared<std::vector<boost::any>>();

    // generate header
    std::stringstream ss;
    ss << "HTTP/1.0 200 OK\r\n"                                        
       << "Content-Type: " << content_type << "\r\n"                 
       << "Content-Length: " << data->size() << "\r\n"                                      
       << "Connection: Keep-Alive\r\n" 
       //<< "Connection: Close\r\n" 
       << "\r\n";
    keep_alive_data->header = ss.str();
    
    // generate footer
    keep_alive_data->footer = "\r\n\r\n";

    // put everything in buffers
    keep_alive_data->buffers.push_back(boost::asio::buffer(keep_alive_data->header));
    keep_alive_data->buffers.push_back(boost::asio::buffer(keep_alive_data->data->data(), keep_alive_data->data->size()));
    keep_alive_data->buffers.push_back(boost::asio::buffer(keep_alive_data->footer));

    boost::asio::async_write( *socket,
                              keep_alive_data->buffers,
                              strand.wrap(boost::bind(
                                            &webserver::dont_close_socket,
                                            this,
                                            keep_alive_data)));

}

void
webserver::send_data(boost::shared_ptr<tcp::socket> socket,
                     const char* content_type,
                     const char* data,
                     size_t data_size)
{
   
    // store all the data that we need to keep alive
    boost::shared_ptr<send_data_data> keep_alive_data = boost::make_shared<send_data_data>();

    // generate header
    std::stringstream ss;
    ss << "HTTP/1.0 200 OK\r\n"                                        
       << "Content-Type: " << content_type << "\r\n"                 
       << "Content-Length: " << data_size << "\r\n"                                      
       << "Connection: Keep-Alive\r\n" 
//       << "Connection: Close\r\n" 
       << "\r\n";
    keep_alive_data->header = ss.str();
    
    // generate footer
    keep_alive_data->footer = "\r\n\r\n";
    
    // put everything in buffers
    keep_alive_data->buffers.push_back(boost::asio::buffer(keep_alive_data->header));
    keep_alive_data->buffers.push_back(boost::asio::buffer(data, data_size));
    keep_alive_data->buffers.push_back(boost::asio::buffer(keep_alive_data->footer));

    boost::asio::async_write( *socket,
                              keep_alive_data->buffers,
                              strand.wrap(boost::bind(
                                            &webserver::dont_close_socket,
                                            this,
                                            keep_alive_data)));

}


static std::vector<long>
parse_request(std::string uri)
{

    // create empty vector
    std::vector<long> ret;

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

    // create result vector
    ret.push_back(zoom_raw);
    ret.push_back(pos_x_raw);
    ret.push_back(pos_y_raw);

    return ret;

}

void
webserver::process_request(boost::shared_ptr<tcp::socket> socket,
                           std::string filename)
{

    //std::cout << "process_request: " << filename << std::endl;

    // send main website if requested
    if(filename == "/")
    {
        send_data(socket, "text/html; charset=utf-8",
                  mandelbrot_html, mandelbrot_html_len); 
        return;
    }

    // send favicon if requested
    if(filename == "/favicon.ico")
    {
        send_data(socket, "image/x-icon",
                  mandelbrot_ico, mandelbrot_ico_len);
        return;
    }

    // if filename is empty, send "bad request"
    if(filename == "")
    {
        send_bad_request_and_close(socket);
        return;
    }

    // parse the filename to see if it is an image request
    std::vector<long> img_coords = parse_request(filename);

    // send 'not found' if it is not an image request
    if(img_coords.size() != 3)
    {
        send_not_found_and_close(socket);
        return;
    }

    // create a new request
    boost::shared_ptr<request> img_request = boost::make_shared<request>();
   
    // set coords
    img_request->zoom = img_coords[0];
    img_request->posx = img_coords[1];
    img_request->posy = img_coords[2];

    // set user-ip
    img_request->user_ip = socket->remote_endpoint().address().to_string();

    // set aborted callback
    img_request->abort = strand.wrap(boost::bind(
                                        &webserver::send_server_error_and_close,
                                        this,
                                        socket));

    // set stillValid callback
    img_request->stillValid = boost::bind(
                                        &webserver::is_socket_still_connected,
                                        this,
                                        socket); 

    // set done callback
    img_request->done = strand.wrap(boost::bind(
                                        &webserver::send_data,
                                        this,
                                        socket,
                                        "image/png",
                                        _1));

    num_requests ++;
    // submit the request
    req_handler->submit_request(img_request);

}


bool
webserver::is_socket_still_connected(boost::shared_ptr<tcp::socket> socket)
{

    return socket->is_open();

}

void
webserver::new_line_read_callback(
                               const boost::system::error_code & error,
                               size_t bytes_transferred,
                               boost::shared_ptr<tcp::socket> socket,
                               boost::shared_ptr<boost::asio::streambuf> buffer,
                               size_t lines_read,
                               std::string requested_file)
{

    // check for errors
    if(error)
    {
        std::cout << error.message() << std::endl;
        // on error, close socket and return without querying the next line read
        socket->close();
        return;
    }

    // create an input stream from buffer
    std::istream instream(&(*buffer));

    // create a string to hold the read line
    std::string line;
    
    // read data to the string
    std::getline(instream, line);
     
    // remove whitespace at front and end of string
    boost::algorithm::trim(line);

    // if this is the first line, read query information from it
    if(lines_read == 0)
    {

        // read filename from line
        std::string filename = read_filename_from_request(line);
        if(filename == "")
        {
            // send error
            send_bad_request_and_close(socket);
            return;
        }

        // remember filename
        requested_file = filename;

    }
    // if this is the last line, process the request
    else if (line.size() == 0)
    {
        // process the request
        process_request(socket, requested_file); 

        // continue reading lines, could be persistent socket
        start_reading_line_from_socket(buffer, socket, 0, "");

        return;
    }

    // query the next line
    start_reading_line_from_socket(buffer, socket, lines_read + 1, requested_file);

}

void
webserver::start_reading_line_from_socket(
                              boost::shared_ptr<boost::asio::streambuf> buffer,
                              boost::shared_ptr<tcp::socket> socket,
                              size_t num_lines_already_read,
                              std::string requested_file)
{

    // connect the buffer to the stream and register data callback
    boost::asio::async_read_until(*socket, *buffer, "\n",
                       strand.wrap(
                       boost::bind(&webserver::new_line_read_callback,
                                   this,
                                   boost::asio::placeholders::error,
                                   boost::asio::placeholders::bytes_transferred,
                                   socket,
                                   buffer,
                                   num_lines_already_read,
                                   requested_file)
                       ));

}

void
webserver::new_connection_callback(const boost::system::error_code & error,
                                   boost::shared_ptr<tcp::socket> socket)
{
    // start listening for another connection, to enable multiple connections
    // at once
    start_listening_for_new_connection();

    // check for error
    if(error)
    {
        // print error message
        std::cerr << "Webserver: Error while waiting for new connection: "
                  << error.message() << std::endl; 

        // drop this connection
        return;
    }

    // create a new buffer for the tcp data stream
    boost::shared_ptr<boost::asio::streambuf> buffer = 
                                   boost::make_shared<boost::asio::streambuf>();

    // connect buffer to socket and start reading
    start_reading_line_from_socket(buffer, socket, 0, "");

}

void
webserver::start_listening_for_new_connection()
{

    // create new socket
    boost::shared_ptr<tcp::socket> connected_socket = 
                                    boost::make_shared<tcp::socket>(io_service);

    // register socket and callback, to wait for new connection
    acceptor.async_accept(*connected_socket,
                          strand.wrap(
                          boost::bind(&webserver::new_connection_callback,
                                      this,
                                      boost::asio::placeholders::error,
                                      connected_socket)));

}

void
webserver::external_start(hpx::runtime* rt)
{
    // register this thread to hpx
    registration_wrapper wrap(rt, "asio_webserver");

    // register callback for new connections
    start_listening_for_new_connection();
    
    // start the io_service
    io_service.run(); 

}

void
webserver::start()
{

    // forward this call to an external os thread 
    asio_thread = boost::thread(hpx::util::bind(&webserver::external_start,
                                                this,
                                                hpx::get_runtime_ptr()));

}

void
webserver::stop()
{

    // stop the io service
    io_service.stop();

    // wait for the operating system thread to finish
    asio_thread.join();

}
