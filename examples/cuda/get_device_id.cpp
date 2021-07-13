// Copyright (c)       2021 Pedro Barbosa
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/future.hpp>

#include <hpxcl/cuda.hpp>

using namespace hpx::cuda;

int main(int argc, char* argv[]) {

	// Get list of available Cuda Devices.
	std::vector<device> devices = get_all_devices(2, 0).get();
	
	// Check whether there are any devices
	if (devices.size() < 1) {
		hpx::cerr << "No CUDA devices found!" << hpx::endl;
		return hpx::finalize();
	}

	// Create a device component from the first device found
	device cudaDevice_0 = devices[0];

	// Create a buffer
	buffer test_buffer_0 = cudaDevice_0.create_buffer(sizeof(int)).get();

	// Get buffer parent id
	int device_id_0 = test_buffer_0.get_device_id().get();
	std::cout << device_id_0 << std::endl;

	// Comment the following section in case there's only one device
	// Create a device component from the second device found
	device cudaDevice_1 = devices[1];
	
	// Create a buffer
	buffer test_buffer_1 = cudaDevice_1.create_buffer(sizeof(int)).get();

	// Get buffer parent id
	int device_id_1 = test_buffer_1.get_device_id().get();
	std::cout << device_id_1 << std::endl;

	return EXIT_SUCCESS;
}


