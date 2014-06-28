// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MANDELBROT_MANDELBROTWORKER_BUFFERMANAGER_H_
#define MANDELBROT_MANDELBROTWORKER_BUFFERMANAGER_H_

#include "../../../opencl.hpp"

#include <map>

/* 
 * a worker.
 * will ask the workqueue for new work until the workqueue finishes.
 * this is the only class that actually uses the hpxcl.
 */
class mandelbrotworker_buffermanager
{

    public:
        // initializes the buffermanager
        mandelbrotworker_buffermanager(hpx::opencl::device device_,
                                       size_t initial_buffer_size,
                                       bool verbose,
                                       cl_mem_flags memflags);

        // get a buffer
        hpx::opencl::buffer
        get_buffer(size_t buffersize);

    // private functions
    private:
        void allocate_buffer(size_t size);


    // private attributes
    private:
        hpx::opencl::device device;
        typedef std::map<size_t, hpx::opencl::buffer> buffer_map_type;
        buffer_map_type buffers;
        bool verbose;
        cl_mem_flags memflags;
};

#endif

