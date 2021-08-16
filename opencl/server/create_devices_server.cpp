// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The Header
#include "create_devices.hpp"

// HPXCL tools
#include "../tools.hpp"

// HPXCL dependencies
#include "../device.hpp"
#include "device.hpp"

// Other dependencies
#include <vector>
#include <string>

///////////////////////////////////////////////////
/// Local functions
///
static std::vector<int> parse_version_string(std::string version_str) {
  try {
    // Make sure the version string starts with "OpenCL "
    HPX_ASSERT(version_str.compare(0, 7, "OpenCL ") == 0);

    // Cut away the "OpenCL " in front of the version string
    version_str = version_str.substr(7);

    // Cut away everything behind the version number
    version_str = version_str.substr(0, version_str.find(" "));

    // Get major version string
    std::string version_str_major =
        version_str.substr(0, version_str.find("."));

    // Get minor version string
    std::string version_str_minor =
        version_str.substr(version_str_major.size() + 1);

    // create output vector
    std::vector<int> version_numbers(2);

    // Parse version number
    version_numbers[0] = ::atoi(version_str_major.c_str());
    version_numbers[1] = ::atoi(version_str_minor.c_str());

    // Return the parsed version number
    return version_numbers;

  } catch (const std::exception &) {
    hpx::cerr << "Error while parsing OpenCL Version!" << hpx::endl;
    std::vector<int> version_numbers(2);
    version_numbers[0] = -1;
    version_numbers[1] = 0;
    return version_numbers;
  }
}

///////////////////////////////////////////////////
/// Implementations
///

// This method initializes the devices-list if it's not done yet.
std::vector<hpx::opencl::device> hpx::opencl::server::create_devices(
    cl_device_type device_type, std::string min_cl_version) {
  HPX_ASSERT(hpx::opencl::tools::runs_on_medium_stack());

  // Parse required OpenCL version
  std::vector<int> required_version = parse_version_string(min_cl_version);

  // Initialize device list
  std::vector<hpx::opencl::device> devices;

  // Declaire the cl error code variable
  cl_int err;

  // Query for number of available platforms
  cl_uint num_platforms;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  cl_ensure(err, "clGetPlatformIDs()");

  // Retrieve platforms
  std::vector<cl_platform_id> platforms(num_platforms);
  err = clGetPlatformIDs(num_platforms, platforms.data(), NULL);
  cl_ensure(err, "clGetPlatformIDs()");

  // Search on every platform
  for (const auto &platform : platforms) {
    // Query for number of available devices
    cl_uint num_devices_on_platform;
    err = clGetDeviceIDs(platform, device_type, 0, NULL,
                         &num_devices_on_platform);
    if (err == CL_DEVICE_NOT_FOUND) continue;
    cl_ensure(err, "clGetDeviceIDs()");

    // Retrieve devices
    std::vector<cl_device_id> devices_on_platform(num_devices_on_platform);
    err = clGetDeviceIDs(platform, device_type, num_devices_on_platform,
                         devices_on_platform.data(), NULL);
    cl_ensure(err, "clGetDeviceIDs()");

    // Add devices_on_platform to devices
    for (const auto &device : devices_on_platform) {
      // Get OpenCL Version string length
      std::size_t version_string_length;
      err = clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, NULL,
                            &version_string_length);
      cl_ensure(err, "clGetDeviceInfo()");

      // Get OpenCL Version string
      std::vector<char> version_string_arr(version_string_length);
      err = clGetDeviceInfo(device, CL_DEVICE_VERSION, version_string_length,
                            version_string_arr.data(), NULL);
      cl_ensure(err, "clGetDeviceInfo()");

      // Convert to std::string
      std::size_t length = 0;
      while (length < version_string_arr.size()) {
        if (version_string_arr[length] == '\0') break;
        length++;
      }
      std::string version_string(version_string_arr.begin(),
                                 version_string_arr.begin() + length);

      // Parse
      std::vector<int> version = parse_version_string(version_string);

#ifndef HPXCL_ALLOW_OPENCL_1_0_DEVICES

      // only allow machines with version 1.1 or higher
      if (version[0] < 1) continue;
      if (version[0] == 1 && version[1] < 1) continue;

#endif  // HPXCL_ALLOW_OPENCL_1_0_DEVICES

      // Check if device supports required version
      if (version[0] < required_version[0]) continue;
      if (version[0] == required_version[0]) {
        if (version[1] < required_version[1]) continue;
      }

      // Create a new device client
      hpx::opencl::device device_client(
          hpx::components::new_<hpx::opencl::server::device>(hpx::find_here()));

      // Initialize device server locally
      std::shared_ptr<hpx::opencl::server::device> device_server =
          hpx::get_ptr<hpx::opencl::server::device>(device_client.get_id())
              .get();
      device_server->init(device);

      // Add device to list of valid devices
      devices.push_back(device_client);
    }
  }

  return devices;
}
