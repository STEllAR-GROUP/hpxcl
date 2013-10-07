// Copyright (c)	2013 Martin Stumpf
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "device.hpp"
#include "../tools.hpp"

#include <CL/cl.h>


using namespace hpx::opencl::server;

void device::test()
{

	cl_int err;

	cl_uint num_Platforms;

	err = clGetPlatformIDs(0,NULL,&num_Platforms);
	clEnsure(err, "clGetPlatformIDs");

	
	cl_platform_id* platforms = new cl_platform_id[num_Platforms];

	err = clGetPlatformIDs(num_Platforms, platforms, NULL);
	clEnsure(err, "clGetPlatformIDs");

	
	for(cl_uint i = 0; i < num_Platforms; i++)
	{
		char name[128];
		err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, &name, NULL);
		clEnsure(err, "clGetPlatformIDs");
		hpx::cout << "Platform #" << i << ": " <<  name << hpx::endl;
	}

	delete(platforms);
}

// Constructor
/*device::device(cl_device_id deviceID)
{
	this->deviceID = deviceID;
	
	// Retrieve platformID
	cl_int err;
	err = clGetDeviceInfo(deviceID, CL_DEVICE_PLATFORM, sizeof(platformID), &platformID, NULL);
	clEnsure(err, "clGetDeviceInfo");
}*/
