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

//###########################################################################
//main
//###########################################################################
 
int main(int argc, char*argv[]) {

	if (argc != 3) {
		printf("Usage: %s #m #n\n", argv[0]);
		exit(1);
	}

	int *m,*n,i;
	m = (int *) malloc(sizeof(int));
	n = (int *) malloc(sizeof(int));

	//Initilizing the matrix dimensions
	m[0] = atoi(argv[1]);
	n[0] = atoi(argv[2]);

	double time = 0;

	clock_t begin = clock();

	cl_device_id deviceId = NULL;
	cl_context context = NULL;
	cl_command_queue commandQueue = NULL;

	double *alpha;
	int *count;

	//allocating memory for the vectors
	alpha = (double *) malloc(sizeof(double));
	count = (int *) malloc(sizeof(int));

	double *A,*B,*C, *A_data;
	int *A_indices, *A_pointers;

	//initializing values of alpha and beta
	alpha[0] = 1.0;
	count[0] = 0;

	A = (double *) malloc(m[0]*n[0]*sizeof(double));
    B = (double *) malloc(n[0]*sizeof(double));
    C = (double *) malloc(m[0]*sizeof(double));

    //Input can be anything sparse
	for (i = 0; i < (m[0]*n[0]); i++) {
		if((i%n[0]) == 0){
			A[i] = (double)(i+1);
			count[0]++;
		}
	}

	A_data = (double *) malloc(count[0]*sizeof(double));
	A_indices = (int *) malloc(count[0]* sizeof(int));
	A_pointers = (int *) malloc(m[0]* sizeof(int));

	for (i = 0; i < (1*n[0]); i++) {
		B[i] = (double)(-i-1);
	}

	for (i = 0; i < (m[0]*1); i++) {
		C[i] = 0.0;
	}

	//Counters for compression
	int data_counter = 0;
	int index_counter = 0;
	int pointer_counter = -1;

	//Compressing Matrix A
	for (i = 0; i < (m[0]*n[0]); i++) {
		if(A[i] != 0)
		{
			A_data[data_counter++] = A[i];
			if(((int)i/n[0]) != pointer_counter)
				A_pointers[++pointer_counter] = index_counter;
			A_indices[index_counter++] = (i%n[0]);
		}
	}

	//Memory objects for kernel parameters
	cl_mem ADataMemobj = NULL;
	cl_mem AIndexMemobj = NULL;
	cl_mem APointerMemobj = NULL;
	cl_mem BMemobj = NULL;
	cl_mem CMemobj = NULL;
	cl_mem mMemobj = NULL;
	cl_mem nMemobj = NULL;
	cl_mem countMemobj = NULL;
	cl_mem alphaMemobj = NULL;

	//Some opencl objects
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platformId = NULL;
	cl_uint numDevices;
	cl_uint numPlatforms;
	cl_int ret;
	size_t contextDescriptorSize;

	//reading kernel from file
	FILE *file;
	char fileName[] = "./smvp.cl";
	char *kernelSource;
	size_t sourceSize;
	
	//Load the source code from file
	file = fopen(fileName, "r");
	if(!file) {
		printf("Failed to load the kernel file. \n");
		exit(1);
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
	ADataMemobj = clCreateBuffer(context, CL_MEM_READ_ONLY, count[0] * sizeof(double), A_data, &ret);
	AIndexMemobj = clCreateBuffer(context, CL_MEM_READ_ONLY, count[0]* sizeof(int), A_indices, &ret);
	APointerMemobj = clCreateBuffer(context, CL_MEM_READ_ONLY, m[0]* sizeof(int), A_pointers, &ret);	
	BMemobj = clCreateBuffer(context, CL_MEM_READ_ONLY, n[0] * sizeof(double), B, &ret);
	CMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE, m[0]*n[0] * sizeof(double), C, &ret);
	mMemobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 1 * sizeof(int), m, &ret);
	nMemobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 1 * sizeof(int), n, &ret);
	countMemobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 1 * sizeof(int), count, &ret);
	alphaMemobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 1 * sizeof(double), alpha, &ret);
	
	//Create kernel program from source
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource,(const size_t *)&sourceSize, &ret);

	//Write data to the buffer
	ret = clEnqueueWriteBuffer(commandQueue, ADataMemobj, CL_TRUE, 0, sizeof(double) * count[0], A_data, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, AIndexMemobj, CL_TRUE, 0, sizeof(int) * count[0], A_indices, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, APointerMemobj, CL_TRUE, 0, sizeof(int) * m[0], A_pointers, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, BMemobj, CL_TRUE, 0, sizeof(double) * n[0], B, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, CMemobj, CL_TRUE, 0, sizeof(double) * m[0], C, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, mMemobj, CL_TRUE, 0, 1 * sizeof(int), m, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, nMemobj, CL_TRUE, 0, 1 * sizeof(int), n, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, countMemobj, CL_TRUE, 0, 1 * sizeof(int), count, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, alphaMemobj, CL_TRUE, 0, 1 * sizeof(double), alpha, 0, NULL, NULL);
	
	//Build the kernel program
	ret = clBuildProgram(program, 1, &deviceId, "-I ./", NULL, NULL);

	//Create a opencl kernel
	kernel = clCreateKernel(program, "smvp", &ret);

	//Pass arguments to kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&ADataMemobj);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&AIndexMemobj);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&APointerMemobj);
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&BMemobj);
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&CMemobj);
	ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&mMemobj);
	ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&nMemobj);
	ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&countMemobj);
	ret = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&alphaMemobj);

	// Execute OpenCL kernel in data parallel
    const int TS = 32;
	const size_t local[2] = { TS, 1 };
	const size_t global[2] = { (int)(pow(2,ceil(log(m[0])/log(2)))), 1 };

	//Execute opencl kernel
    ret = clEnqueueNDRangeKernel( commandQueue, kernel, 2, NULL, global, local, 0, 0, 0 );

	//copy the result back
	ret = clEnqueueReadBuffer(commandQueue, CMemobj, CL_TRUE, 0, m[0]*sizeof(double), C, 0, NULL, NULL);

	//Before program termination
	ret = clFlush(commandQueue);
	ret = clFinish(commandQueue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(ADataMemobj);
	ret = clReleaseMemObject(AIndexMemobj);
	ret = clReleaseMemObject(APointerMemobj);
	ret = clReleaseMemObject(BMemobj);
	ret = clReleaseMemObject(CMemobj);
	ret = clReleaseMemObject(mMemobj);
	ret = clReleaseMemObject(nMemobj);
	ret = clReleaseMemObject(countMemobj);
	ret = clReleaseMemObject(alphaMemobj);
	ret = clReleaseCommandQueue(commandQueue);
	ret = clReleaseContext(context);
	 
	free(kernelSource);
	free(A);
	free(B);
	free(C);
	free(A_data);
	free(A_indices);
	free(A_pointers);
	free(m);
	free(n);
	free(alpha);
	free(count);

	clock_t end = clock();
	time += (double)(end - begin) * 1000 / CLOCKS_PER_SEC;
	printf("%lf\n", time);
	
	return 0;
}