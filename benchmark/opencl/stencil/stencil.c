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

//Protypes of function declarations
bool checkStencil(TYPE *in, TYPE *out, TYPE *s, size_t size);
void fillRandomVector(TYPE *matrix, size_t size);

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
	cl_mem countMemobj = NULL;
	cl_mem inMemobj = NULL;
	cl_mem outMemobj = NULL;
	cl_mem sMemobj = NULL;

	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platformId = NULL;
	cl_uint numDevices;
	cl_uint numPlatforms;
	cl_int ret;
	size_t contextDescriptorSize;

	TYPE *outObject;
	TYPE *sObject; 

	//reading kernel from file
	FILE *file;
	char fileName[] = "./stencilKernel.cl";
	char *kernelSource;
	size_t sourceSize;
	
	//Load the source code from file
	file = fopen(fileName, "r");
	if(!file) {
		printf("Failed to load the kernel file. \n");
		exit(1);
	}

	outObject = (TYPE *) malloc(count * sizeof(TYPE));
	sObject = (TYPE *) malloc(3 * sizeof(TYPE));

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

	//randomly generate s vector
	sObject[0] = 0.5;
	sObject[1] = 1.0;
	sObject[2] = 0.5;

	//Create memory object
	countMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(size_t), NULL, &ret);
	inMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE,count * sizeof(TYPE), NULL, &ret);
	outMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE,count * sizeof(TYPE), NULL, &ret);
	sMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE,3 * sizeof(TYPE), NULL, &ret);

	//Create kernel program from source
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource,(const size_t *)&sourceSize, &ret);

	//Build the kernel program
	ret = clBuildProgram(program, 1, &deviceId, "-I ./", NULL, NULL);

	//Create a opencl kernel
	kernel = clCreateKernel(program, "stencil", &ret);

	//Pass arguments to kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&countMemobj);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&inMemobj);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&outMemobj);
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&sMemobj);

	//Execute opencl kernel
	ret = clEnqueueTask(commandQueue, kernel, 0, NULL,NULL);

	//copy the result back
	ret = clEnqueueReadBuffer(commandQueue, outMemobj, CL_TRUE, 0, count * sizeof(TYPE),outObject, 0, NULL, NULL);

	//Before program termination
	ret = clFlush(commandQueue);
	ret = clFinish(commandQueue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(countMemobj);
	ret = clReleaseMemObject(inMemobj);
	ret = clReleaseMemObject(outMemobj);
	ret = clReleaseMemObject(sMemobj);
	ret = clReleaseCommandQueue(commandQueue);
	ret = clReleaseContext(context);
	 
	free(kernelSource);
	free(outObject);

	return 0;
}

bool checkStencil(TYPE *in, TYPE *out, TYPE *s, size_t size) {
	size_t i;
	bool validate = true;

	for (i = 1; i < size - 1; ++i) {
		TYPE res = in[i - 1] * s[0] + in[i] * s[1] + in[i + 1] * s[2];

		if (abs(res - out[i]) >= 10e-5) {
			validate = false;
			break;
		}
	}

	return validate;
}

void fillRandomVector(TYPE *matrix, size_t size) {
	size_t i;
	srand(time(NULL));

	for (i = 0; i < size; i++) {

		matrix[i] = (TYPE) (0.5) * ((TYPE) rand()) / (TYPE) RAND_MAX;
	}

}
