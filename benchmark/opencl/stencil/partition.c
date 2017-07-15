// Copyright (c)       2017 Madhavan Seshadri
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#include <CL/cl.h>

#include "stencil.h"

//add this line for compiling with Visual Studio 
#pragma comment(lib, "OpenCL.lib")

//###########################################################################
//Properties for reading the kernel
//###########################################################################

#define SOURCE_SIZE_MAX (0x100000)
bool checkKernel(TYPE *in, size_t size);
//###########################################################################
//main
//###########################################################################
 
int main(int argc, char*argv[]) {

	if(argc != 2)
	{
		printf("Usage: %s #elements.\n", argv[0]);
		exit(1);
	}

	size_t count = atoi(argv[1]);

	cl_device_id deviceId = NULL;
	cl_context context = NULL;
	cl_command_queue commandQueue = NULL;

	//Memory objects for kernel parameters
	cl_mem inMemobj = NULL;
	cl_mem offsetMemobj = NULL;

	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platformId = NULL;
	cl_uint numDevices;
	cl_uint numPlatforms;
	cl_int ret;
	size_t contextDescriptorSize;

	const int blockSize = 256;
	const int nStreams = 4;
	const int n = pow(2,count) * 1024 * blockSize * nStreams;
	const int streamSize = n/nStreams;
	const int streamBytes = streamSize * sizeof(TYPE);
	const int bytes = n * sizeof(TYPE);
	int i;
	int offset;

	//reading kernel from file
	FILE *file;
	char fileName[] = "./stencilKernel.cl";
	char *kernelSource;
	size_t sourceSize;

	TYPE *in;

	//Load the source code from file
	file = fopen(fileName, "r");
	if(!file) {
		printf("Failed to load the kernel file. \n");
		exit(1);
	}

	in = (TYPE *) malloc(bytes);

	kernelSource = (char *) malloc(SOURCE_SIZE_MAX);
	sourceSize = fread(kernelSource, 1, SOURCE_SIZE_MAX, file);
	fclose(file);

	//Get platform information
	ret = clGetPlatformIDs(1, &platformId, &numPlatforms);

	//get list of devices
	ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId, &numDevices);
	
	//create opencl device context
	context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &ret);

	//get device context
	clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, 0, &contextDescriptorSize); 

	//command queue for the first device
	commandQueue = clCreateCommandQueue(context, deviceId, 0, &ret);

	//Create kernel program from source
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource,(const size_t *)&sourceSize, &ret);

	//Build the kernel program
	ret = clBuildProgram(program, 1, &deviceId, "-I ./", NULL, NULL);

	for(i = 0; i < nStreams; i++)
	{
		offset = i * streamSize;
	
		//Create a opencl kernel
		kernel = clCreateKernel(program, "partition", &ret);
		
		inMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE, streamBytes, NULL, &ret);
		offsetMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(size_t), &offset, &ret);
		
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&offsetMemobj);
		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&inMemobj);
		
		//Execute opencl kernel
		ret = clEnqueueTask(commandQueue, kernel, 0, NULL,NULL);

		//copy the result back
		ret = clEnqueueReadBuffer(commandQueue, inMemobj, CL_TRUE, 0, streamBytes, &in[offset], 0, NULL, NULL);

		ret = clFlush(commandQueue);
		ret = clReleaseKernel(kernel);
		ret = clReleaseMemObject(inMemobj);
		ret = clReleaseMemObject(offsetMemobj);
	}

	printf("Validate Result: %d", checkKernel(in,n));

	//Before program termination
	ret = clFlush(commandQueue);
	ret = clFinish(commandQueue);
	ret = clReleaseProgram(program);
	ret = clReleaseCommandQueue(commandQueue);
	ret = clReleaseContext(context);

	free(in);

	return 0;
}

bool checkKernel(TYPE *in, size_t size) {
	bool validate = true;
	size_t i;
	for (i = 0; i < size ; i++) {

		TYPE error = abs(in[i]-1.0f);

		if (error > 10e-5) { validate = false; break;}
	}

	return validate;
}
