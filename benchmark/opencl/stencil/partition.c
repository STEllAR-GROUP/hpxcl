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

void checkBuild(int err, cl_program program, cl_device_id deviceId){
	 if (err != CL_SUCCESS) {
	char *buff_erro;
	cl_int errcode;
	size_t build_log_len;
	errcode = clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
	if (errcode) {
            printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
            exit(-1);
        }

    buff_erro = malloc(build_log_len);
    if (!buff_erro) {
        printf("malloc failed at line %d\n", __LINE__);
        exit(-2);
    }

    errcode = clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, build_log_len, buff_erro, NULL);
    if (errcode) {
        printf("clGetProgramBuildInfo failed at line %d\n", __LINE__);
        exit(-3);
    }

    fprintf(stderr,"Build log: \n%s\n", buff_erro); //Be careful with  the fprint
    free(buff_erro);
	}
}


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
	cl_mem streamSizeMemobj = NULL;

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
	int stream = streamSize;
	const size_t global_work_size = n;
	const size_t local_work_size = 256;

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
	commandQueue = clCreateCommandQueue(context, deviceId, CL_QUEUE_PROFILING_ENABLE, &ret);

	//Create kernel program from source
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource,(const size_t *)&sourceSize, &ret);

	//Build the kernel program
	ret = clBuildProgram(program, 1, &deviceId, "-I ./", NULL, NULL);
	checkBuild(ret, program, deviceId);

	for(i = 0; i < nStreams; i++)
	{
		offset = i * streamSize;
	
		//Create a opencl kernel
		kernel = clCreateKernel(program, "partition", &ret);
		
		inMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE, streamBytes, NULL, &ret);
		offsetMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(size_t), &offset, &ret);
		streamSizeMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), &stream, &ret);

		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&offsetMemobj);
		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&inMemobj);
		ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&streamSizeMemobj);

		cl_event event = NULL;
		cl_ulong time_start = 0, time_end = 0;
	
		//Execute opencl kernel
		ret = clEnqueueNDRangeKernel (commandQueue, kernel, 1, 0, &global_work_size, &local_work_size, 0, NULL, &event);
		clWaitForEvents(1, &event);

	    clFinish(commandQueue);
	    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

		//copy the result back
		ret = clEnqueueReadBuffer(commandQueue, inMemobj, CL_TRUE, 0, streamBytes, &in[offset], 0, NULL, NULL);
		ret = clFinish(commandQueue);
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
