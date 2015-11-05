// Copyright (c)    2015 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Header File
#include <hpx/include/components.hpp>

#include "cuda/server/get_devices.hpp"
#include "cuda/cuda_error_handling.hpp"

#include <vector>

namespace hpx {

namespace cuda {

namespace server {

std::vector<hpx::cuda::device> get_devices(int major, int minor)
{
    std::vector<hpx::cuda::device> devices;

    int count = 0;

    cudaGetDeviceCount(&count);
    checkCudaError("get_devices");

    for (int device_id = 0; device_id < count; ++device_id)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        checkCudaError("get_devices");

        if (prop.major >= major && prop.minor >= minor)
            devices.push_back(hpx::cuda::device(find_here(), device_id));
    }

    return devices;
}

}
}
}

HPX_PLAIN_ACTION(
    hpx::cuda::server::get_devices,
    hpx_cuda_server_get_devices_action);

