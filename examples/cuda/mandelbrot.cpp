// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/hpx_main.hpp>

#include <hpxcl/cuda.hpp>

using namespace hpx::cuda;

//###########################################################################
//Kernels
//###########################################################################

static const char kernel_src[] = "";

//###########################################################################
//Main
//###########################################################################

int main(int argc, char* argv[]){

	//Reading a list of available devices in hpx locality
	//Returns a future
	std::std::vector<device> devices = get_all_devices(2, 0).get();

	if(devices.size() < 1) {
		hpx::cerr << "No CUDA devices found! Terminating...." << hpx::endl;
		return hpx::finalize();
	}
	



	return EXIT_SUCCESS;
}