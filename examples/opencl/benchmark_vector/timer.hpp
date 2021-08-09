// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BENCHMARK_TIMER_H_
#define BENCHMARK_TIMER_H_

#include <boost/date_time/posix_time/posix_time.hpp>

static boost::posix_time::ptime start_time;

static void timer_start() {
  // Measure start time
  start_time = boost::posix_time::microsec_clock::local_time();
}

static double timer_stop() {
  // Measure stop time
  boost::posix_time::ptime stop_time =
      boost::posix_time::microsec_clock::local_time();

  // Calculate difference
  boost::posix_time::time_duration diff = stop_time - start_time;

  return diff.total_microseconds() / 1000.0;
}

#endif  // BENCHMARK_TIMER_H_
