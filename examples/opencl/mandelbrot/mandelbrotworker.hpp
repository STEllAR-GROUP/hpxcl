// Copyright (c)       2014 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef MANDELBROT_MANDELBROTWORKER_H_
#define MANDELBROT_MANDELBROTWORKER_H_

#include <hpxcl/opencl.hpp>

#include <boost/function.hpp>
#include <boost/bind.hpp>

#include "workload.hpp"
#include "work_queue.hpp"

#include <hpx/lcos/local/event.hpp>

#include <memory>

/*
 * a worker.
 * will ask the workqueue for new work until the workqueue finishes.
 * this is the only class that actually uses the hpxcl.
 */
class mandelbrotworker {
 public:
  // initializes the worker
  mandelbrotworker(
      hpx::opencl::device device_, size_t num_workers,
      boost::function<bool(std::shared_ptr<workload>*)> request_new_work,
      boost::function<void(std::shared_ptr<workload>&)> deliver_done_work,
      bool verbose, size_t workpacket_size_hint_x,
      size_t workpacket_size_hint_y);

  // waits for the worker to finish
  void join();

  // waits for the worker to finish initialization
  void wait_for_startup_finished();

  // destructor, basically waits for the worker to finish
  ~mandelbrotworker();

 private:
  // the main worker function, runs the main work loop
  size_t worker_main(hpx::opencl::kernel precalc_kernel,
                     hpx::opencl::kernel kernel, size_t workpacket_size_hint_x,
                     size_t workpacket_size_hint_y);

  // the startup function, initializes the kernel and starts the workers
  void worker_starter(size_t num_workers, size_t workpacket_size_hint_x,
                      size_t workpacket_size_hint_y);

  // private attributes
 private:
  const bool verbose;
  const unsigned int id;
  hpx::opencl::device device;
  hpx::lcos::shared_future<void> worker_finished;
  std::shared_ptr<hpx::lcos::local::event> worker_initialized;
  boost::function<bool(std::shared_ptr<workload>*)> request_new_work;
  boost::function<void(std::shared_ptr<workload>&)> deliver_done_work;
};

#endif
