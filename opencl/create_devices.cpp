// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Header File
#include "create_devices.hpp"

// Internal Dependencies
#include "device.hpp"
#include "server/create_devices.hpp"

// HPX dependencies
#include <hpx/lcos/when_all.hpp>

static hpx::lcos::future<std::vector<hpx::opencl::device>>
create_devices_on_nodes(std::vector<hpx::naming::id_type> &&localities,
                        cl_device_type device_type,
                        std::string required_cl_version) {
  // query all devices
  std::vector<hpx::lcos::future<std::vector<hpx::opencl::device>>>
      locality_device_futures;
  for (auto &locality : localities) {
    // get all devices on locality
    hpx::lcos::future<std::vector<hpx::opencl::device>> locality_device_future =
        hpx::opencl::create_devices(locality, device_type, required_cl_version);

    // add locality device future to list of futures
    locality_device_futures.push_back(std::move(locality_device_future));
  }

  // combine futures
  hpx::lcos::future<
      std::vector<hpx::lcos::future<std::vector<hpx::opencl::device>>>>
      combined_locality_device_future = hpx::when_all(locality_device_futures);

  // create result future
  hpx::lcos::future<std::vector<hpx::opencl::device>> result_future =
      combined_locality_device_future.then(hpx::util::bind(

          // define combining function inline
          [](hpx::lcos::future<
              std::vector<hpx::lcos::future<std::vector<hpx::opencl::device>>>>
                 parent_future) -> std::vector<hpx::opencl::device> {
            // initialize the result list
            std::vector<hpx::opencl::device> devices;

            // get vector from parent future
            std::vector<hpx::lcos::future<std::vector<hpx::opencl::device>>>
                locality_device_futures = parent_future.get();

            // for each future, take devices out and join in one list
            for (auto &locality_device_future : locality_device_futures) {
              // wait for device query to finish
              std::vector<hpx::opencl::device> locality_devices =
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

hpx::lcos::future<std::vector<hpx::opencl::device>> hpx::opencl::create_devices(
    hpx::naming::id_type node_id, cl_device_type device_type,
    std::string required_cl_version) {
  typedef hpx::opencl::server::create_devices_action action;
  return async<action>(node_id, device_type, required_cl_version);
}

hpx::lcos::future<std::vector<hpx::opencl::device>>
hpx::opencl::create_local_devices(cl_device_type device_type,
                                  std::string required_cl_version) {
  // get local locality id
  hpx::naming::id_type locality = hpx::find_here();

  // find devices on localities
  return create_devices(locality, device_type, required_cl_version);
}

hpx::lcos::future<std::vector<hpx::opencl::device>>
hpx::opencl::create_remote_devices(cl_device_type device_type,
                                   std::string required_cl_version) {
  // get remote HPX localities
  std::vector<hpx::naming::id_type> localities = hpx::find_remote_localities();

  // find devices on localities
  return create_devices_on_nodes(std::move(localities), device_type,
                                 required_cl_version);
}

hpx::lcos::future<std::vector<hpx::opencl::device>>
hpx::opencl::create_all_devices(cl_device_type device_type,
                                std::string required_cl_version) {
  // get all HPX localities
  std::vector<hpx::naming::id_type> localities = hpx::find_all_localities();

  // find devices on localities
  return create_devices_on_nodes(std::move(localities), device_type,
                                 required_cl_version);
}
