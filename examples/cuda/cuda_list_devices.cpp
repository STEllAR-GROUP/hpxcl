// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>

#include <hpxcl/cuda.hpp>
#include <sstream>

// hpx_main, is the actual main called by hpx
int hpx_main(int argc, char* argv[]) {
	{

		//Get list of available CUDA Devices.
		std::vector<hpx::cuda::device> devices = hpx::cuda::get_all_devices(1,
				0).get();

		// Check whether there are any devices
		if (devices.size() < 1) {
			hpx::cerr << "No CUDA devices found!" << hpx::endl;
			return hpx::finalize();
		}

		hpx::cout << hpx::endl << "Devices:" << hpx::endl << hpx::endl;

		// print a lot of information about every device
		for (auto &device : devices) {

			device.get_cuda_info();
			// add newline before starting a new device
			hpx::cout << hpx::endl;

		}
	}

// End the program
	return hpx::finalize();
}
