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

//add this line for compiling with Visual Studio 
#pragma comment(lib, "OpenCL.lib")

//###########################################################################
//Properties for reading the kernel
//###########################################################################

#define SOURCE_SIZE_MAX (0x100000)

///////////////////////////////////////////////////////////////////////////////
double mysecond() {
	clock_t time;
	time = clock();
	return ((double) time);
}

//###########################################################################
//stream benchmark
//###########################################################################

int stream_benchmark(int size, int iterations) {

	cl_device_id deviceId = NULL;
	cl_context context = NULL;
	cl_command_queue commandQueue = NULL;

	//Memory objects for kernel parameters
	cl_mem aMemobj = NULL;
	cl_mem bMemobj = NULL;
	cl_mem cMemobj = NULL;
	cl_mem sMemobj = NULL;
	cl_mem scaleMemobj = NULL;
	cl_mem sizeMemobj = NULL;

	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platformId = NULL;
	cl_uint numDevices;
	cl_uint numPlatforms;
	cl_int ret;
	size_t contextDescriptorSize;

	//Define vectors to be used by the kernel
	double* a;
	double* b;
	double* c;
	size_t* s;
	double* scale;
	double quantum;
	int iteration;

	//reading kernel from file
	FILE *file;
	char fileName[] = "./streamKernel.cl";
	char *kernelSource;
	size_t sourceSize;
	
	//Load the source code from file
	file = fopen(fileName, "r");
	if(!file) {
		printf("Failed to load the kernel file. \n");
		exit(1);
	}

	a = malloc(a, sizeof(double) * size);
	b = malloc(a, sizeof(double) * size);
	c = malloc(a, sizeof(double) * size);
	scale = malloc(scale, sizeof(double));
	s = malloc(s, sizeof(size_t));

	s[0] = size;
	scale[0] = 2.0;

	//Initialize all the arrays
	for(size_t i = 0 ; i < size ; i++)
	{
		a[i] = 1.0;
		b[i] = 2.0;
		c[i] = 0.0;
	}

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

	//Create memory object
	aMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(double) * size, a, &ret);
	bMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(double) * size, b, &ret);
	cMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(double) * size, c, &ret);
	sMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(size_t) , s, &ret);
	scaleMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(double), scale, &ret);
	sizeMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(int), &size, &ret);

	//Create kernel program from source
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource,(const size_t *)&sourceSize, &ret);

	//Build the kernel program
	ret = clBuildProgram(program, 1, &deviceId, "-I ./", NULL, NULL);

	//Create a opencl kernel
	kernel = clCreateKernel(program, "STREAM_Scale", &ret);

	//Pass arguments to kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&aMemobj);
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bMemobj);
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&scaleMemobj);
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&sizeMemobj);


	double time = mysecond();
	//Execute opencl kernel
	ret = clEnqueueTask(commandQueue, kernel, 0, NULL,NULL);
	time = 1.0E6 * (mysecond() - time);

	quantum = checktick();
	if (quantum >= 1) {
		printf("Clock precision is %lf microseconds.\n", quantum);
	} else {
		printf("Clock precision is less than one microsecond.\n", quantum);
		quantum = 1;
	}

	printf("Each test below will take on the order of %lf microseconds. (= %lf clock ticks)\nIncrease the size of the arrays if this shows that you are not getting at least 20 clock ticks per test.\n", time, (time/quantum));
	printf("WARNING -- The above is only a rough guideline.\nFor best results, please be sure you know the precision of your system timer.\n");

	ret = clReleaseKernel(kernel);
	ret = clReleaseMemObject(scaleMemobj);
	ret = clFlush(commandQueue);

	scale[0] = 3.0;
	scaleMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(double), scale, &ret);

	double timing[4][iterations];
	for (iteration = 0; iteration != iterations; ++iteration) {

		// Copy
		timing[0][iteration] = mysecond();

		//Create a opencl kernel
		kernel = clCreateKernel(program, "STREAM_Copy", &ret);

		//Pass arguments to kernel
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&aMemobj);
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bMemobj);
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&sizeMemobj);

		ret = clEnqueueTask(commandQueue, kernel, 0, NULL,NULL);
		timing[0][iteration] = mysecond() - timing[0][iteration];

		ret = clFlush(commandQueue);
		ret = clReleaseKernel(kernel);

		//Scale
		timing[1][iteration] = mysecond();

		//Create a opencl kernel
		kernel = clCreateKernel(program, "STREAM_Scale", &ret);

		//Pass arguments to kernel
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&aMemobj);
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bMemobj);
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&scaleMemobj);
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&sizeMemobj);

		ret = clEnqueueTask(commandQueue, kernel, 0, NULL,NULL);
		timing[1][iteration] = mysecond() - timing[1][iteration];

		ret = clFlush(commandQueue);
		ret = clReleaseKernel(kernel);

		// Add
		timing[2][iteration] = mysecond();

		//Create a opencl kernel
		kernel = clCreateKernel(program, "STREAM_Add", &ret);

		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&aMemobj);
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bMemobj);
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&cMemobj);
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&sizeMemobj);

		ret = clEnqueueTask(commandQueue, kernel, 0, NULL,NULL);
		timing[2][iteration] = mysecond() - timing[2][iteration];

		ret = clFlush(commandQueue);
		ret = clReleaseKernel(kernel);

		// Triad
		timing[3][iteration] = mysecond();

		//Create a opencl kernel
		kernel = clCreateKernel(program, "STREAM_Triad", &ret);

		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&aMemobj);
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bMemobj);
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&cMemobj);
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&scaleMemobj);
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&sizeMemobj);

		ret = clEnqueueTask(commandQueue, kernel, 0, NULL,NULL);
		timing[3][iteration] = mysecond() - timing[3][iteration];

		ret = clFlush(commandQueue);
		ret = clReleaseKernel(kernel);
	}

	//Before program termination
	ret = clFlush(commandQueue);
	ret = clFinish(commandQueue);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(aMemobj);
	ret = clReleaseMemObject(bMemobj);
	ret = clReleaseMemObject(cMemobj);
	ret = clReleaseMemObject(scaleMemobj);
	ret = clReleaseMemObject(sizeMemobj);
	ret = clReleaseCommandQueue(commandQueue);
	ret = clReleaseContext(context);
}

//###########################################################################
//main
//###########################################################################

int main(int argc, char*argv[]) {

	if (argc != 3) {
		printf("Usage: %s #elements #iterations", argv[0]);
		exit(1);
	}

	int size = atoi(argv[1]);
	int iterations = atoi(argv[2]);

	/* --- SUMMARY --- */
	const char *label[4] = {
		"Copy:      ",
		"Scale:     ",
		"Add:       ",
		"Triad:     "
	};

	

	return 0;
}
 