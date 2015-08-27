// Copyright (c)    2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Header File
#include "get_devices.hpp"

namespace hpx {

namespace cuda {

namespace server {

// Internal Dependencies
//#include "device.hpp"
//#include "get_devices.hpp"

// HPX dependencies
//#include <hpx/lcos/when_all.hpp>

std::vector<hpx::cuda::device> get_devices(int major, int minor) {

	std::vector<hpx::cuda::device> devices;

	int count;

	cudaGetDeviceCount(&count);
	//checkCudaError();

	for (unsigned int i = 0; i < count; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		//checkCudaError();
		if (prop.major >= major && prop.minor >= minor) {
			hpx::cuda::device device = hpx::cuda::device();
			device.set_device(i);
			devices.push_back(device);
		}
	}

	return devices;
}

}
}
}
HPX_PLAIN_ACTION(hpx::cuda::server::get_devices,hpx_cuda_server_get_devices_action);

