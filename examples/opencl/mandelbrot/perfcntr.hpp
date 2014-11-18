// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPXCL_MANDELBROT_PERFCNTR_HPP_
#define HPXCL_MANDELBROT_PERFCNTR_HPP_

#include <hpx/config.hpp>

#include <vector>

#include <boost/atomic.hpp>
#include <string>
#include <boost/date_time/local_time/local_time.hpp>


namespace hpx { namespace opencl { namespace examples { namespace mandelbrot {

class perfcntr_t
{
    private:
        size_t num_gpus;
        std::vector<std::string> gpu_names;
        boost::atomic<unsigned long> num_pixels_calculated[100];
        boost::posix_time::ptime start_time;

    public:
        perfcntr_t(){
            num_gpus = 0;
        }

        void init(std::vector<std::string> gpu_names);

        void submit(size_t gpuid, unsigned long num_pixels);

        void set_gpu_name(size_t gpuid, std::string gpu_name);

        std::vector<unsigned long> get_counters();

        std::vector<std::string> get_gpu_names();

        long get_current_time();
};

extern perfcntr_t perfcntr;


} } } }

#endif
