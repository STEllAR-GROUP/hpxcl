// Copyright (c)       2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_start.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>

#include "../../cuda.hpp"
#include <sstream>

/*
 static void printinfo(size_t i, size_t j, std::string info_type,
 std::string info)
 {

 hpx::cout << " " << i << "." << j << ". "
 << info_type << ": " << info << hpx::endl;

 }
 */
/*
 static std::string
 device_uint_to_string(cl_uint res)
 {
 std::stringstream ss;
 ss << res;
 return ss.str();
 }
 */

/*
 static std::string
 device_type_to_string(cl_device_type type)
 {

 std::vector<std::string> typelist;

 if(type & CL_DEVICE_TYPE_CPU)
 typelist.push_back("cpu");

 if(type & CL_DEVICE_TYPE_GPU)
 typelist.push_back("gpu");

 if(type & CL_DEVICE_TYPE_ACCELERATOR)
 typelist.push_back("accelerator");

 if(type & CL_DEVICE_TYPE_DEFAULT)
 typelist.push_back("default");

 #ifdef CL_VERSION_1_2
 if(type & CL_DEVICE_TYPE_CUSTOM)
 typelist.push_back("custom");
 #endif

 std::string result = "";

 for(size_t i = 0 ; i < typelist.size(); i++)
 {

 if(i > 0)
 result += ", ";

 result += typelist[i];

 }

 return result;

 }
 */
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
		size_t i = 1;

		for (auto &device : devices) {

			device.get_cuda_info();
			// add newline before starting a new device
			hpx::cout << hpx::endl;

		}
	}

// End the program
	return hpx::finalize();
}

// Main, initializes HPX
int main(int argc, char* argv[]) {

// initialize HPX, run hpx_main
	hpx::start(argc, argv);

// wait for hpx::finalize being called
	return hpx::stop();
}

