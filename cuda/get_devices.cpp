// Copyright (c)    2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Header File
#include <hpx/include/lcos.hpp>

#include "get_devices.hpp"

#include <vector>

static hpx::future<std::vector<hpx::cuda::device>> get_devices_on_nodes(
    std::vector<hpx::naming::id_type> &&localities, int major, int minor) {
  // query all devices
  std::vector<hpx::future<std::vector<hpx::cuda::device>>>
      locality_device_futures;
  for (auto &locality : localities) {
    // get all devices on locality
    hpx::future<std::vector<hpx::cuda::device>> locality_device_future =
        hpx::cuda::get_devices(locality, major, minor);

    // add locality device future to list of futures
    locality_device_futures.push_back(std::move(locality_device_future));
  }

  // combine futures
  hpx::future<std::vector<hpx::future<std::vector<hpx::cuda::device>>>>
      combined_locality_device_future = hpx::when_all(locality_device_futures);

  // create result future
  hpx::future<std::vector<hpx::cuda::device>> result_future =
      combined_locality_device_future.then(hpx::util::bind(

          // define combining function inline
          [](hpx::future<
              std::vector<hpx::future<std::vector<hpx::cuda::device>>>>
                 parent_future) -> std::vector<hpx::cuda::device> {
            // initialize the result list
            std::vector<hpx::cuda::device> devices;

            // get vector from parent future
            std::vector<hpx::future<std::vector<hpx::cuda::device>>>
                locality_device_futures = parent_future.get();

            // for each future, take devices out and join in one list
            for (auto &locality_device_future : locality_device_futures) {
              // wait for device query to finish
              std::vector<hpx::cuda::device> locality_devices =
                  locality_device_future.get();

              // add all devices to device list
              devices.insert(devices.end(), locality_devices.begin(),
                             locality_devices.end());
            }

            return devices;
          },

          hpx::util::placeholders::_1

          ));

  // return the future to the device list
  return result_future;
}

hpx::future<std::vector<hpx::cuda::device>> hpx::cuda::get_devices(
    hpx::naming::id_type node_id, int major, int minor) {
  typedef hpx::cuda::server::get_devices_action action;
  return async<action>(node_id, major, minor);
}

hpx::future<std::vector<hpx::cuda::device>> hpx::cuda::get_local_devices(
    int major, int minor) {
  // get local locality id
  hpx::naming::id_type locality = hpx::find_here();

  // find devices on localities
  return get_devices(locality, major, minor);
}

hpx::future<std::vector<hpx::cuda::device>> hpx::cuda::get_remote_devices(
    int major, int minor) {
  // get remote HPX localities
  std::vector<hpx::naming::id_type> localities = hpx::find_remote_localities();

  // find devices on localities
  return get_devices_on_nodes(std::move(localities), major, minor);
}

hpx::future<std::vector<hpx::cuda::device>> hpx::cuda::get_all_devices(
    int major, int minor) {
  // get all HPX localities
  std::vector<hpx::naming::id_type> localities = hpx::find_all_localities();

  // find devices on localities
  return get_devices_on_nodes(std::move(localities), major, minor);
}
