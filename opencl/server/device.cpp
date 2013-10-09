// Copyright (c)    2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "device.hpp"
#include "../tools.hpp"

#include <CL/cl.h>

using hpx::opencl::clx_device_id;
using namespace hpx::opencl::server;

clx_device_id device::test()
{

    cl_int err;

    cl_uint num_Platforms;

    err = clGetPlatformIDs(0,NULL,&num_Platforms);
    clEnsure(err, "clGetPlatformIDs");

    
    cl_platform_id* platforms = new cl_platform_id[num_Platforms];

    err = clGetPlatformIDs(num_Platforms, platforms, NULL);
    clEnsure(err, "clGetPlatformIDs");

    cl_device_id device = NULL;

    for(cl_uint i = 0; i < num_Platforms; i++)
    {
        char name[128];
        cl_device_type device_type = CL_DEVICE_TYPE_ACCELERATOR | CL_DEVICE_TYPE_GPU;
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, &name, NULL);
        clEnsure(err, "clGetPlatformIDs");
        hpx::cout << "Platform #" << i << ": " <<  name << hpx::endl;
        cl_uint num_Devices;
        err = clGetDeviceIDs(platforms[i], device_type, 0, NULL, &num_Devices);
        if(err == CL_DEVICE_NOT_FOUND) continue;
        clEnsure(err, "clGetDeviceIDs");
        
        cl_device_id *devices = new cl_device_id[num_Devices];
        err = clGetDeviceIDs(platforms[i], device_type, num_Devices, devices, NULL);
        clEnsure(err, "clGetDeviceIDs");

        for(cl_uint j = 0; j < num_Devices; j++)
        {
            err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, &name, NULL);
            clEnsure(err, "clGetDeviceInfo");
            hpx::cout << "\tDevice #" << j << ": " << name << hpx::endl;
            device = devices[j];
        }

        delete(devices);
    }

    delete(platforms);

    return (clx_device_id) device;
}

// Constructor
device::device(clx_device_id deviceID)
{
    this->deviceID = (cl_device_id) deviceID;
    
    // Retrieve platformID
    cl_int err;
    err = clGetDeviceInfo(this->deviceID, CL_DEVICE_PLATFORM, sizeof(platformID), &platformID, NULL);
    clEnsure(err, "clGetDeviceInfo");

}
