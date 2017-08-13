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

	if (argc != 4) {
		printf("Usage: %s #m #n #k\n", argv[0]);
		exit(1);
	}
	int *m,*n,*k,i;
	m = (int *) malloc(sizeof(int));
	n = (int *) malloc(sizeof(int));
	k = (int *) malloc(sizeof(int));

	//Initilizing the matrix dimensions
	m[0] = atoi(argv[1]);
	n[0] = atoi(argv[2]);
	k[0] = atoi(argv[3]);

	double time = 0;

	clock_t begin = clock();
	cl_device_id deviceId = NULL;
	cl_context context = NULL;
	cl_command_queue commandQueue = NULL;

	double *alpha, *beta;

	//allocating memory for the vectors
	
	alpha = (double *) malloc(sizeof(double));
	beta = (double *) malloc(sizeof(double));

    double *A, *B, *C;
	
    A = (double *) malloc(m[0]*k[0]*sizeof(double));
    B = (double *) malloc(k[0]*n[0]*sizeof(double));
    C = (double *) malloc(m[0]*n[0]*sizeof(double));

	//initializing values of alpha and beta
	alpha[0] = 1.0;
	beta[0] = 0.0;

	clock_t end = clock();

	time += (double)(end - begin) / CLOCKS_PER_SEC;

	printf (" Intializing matrix data \n\n");
	begin = clock();
	for (i = 0; i < (m[0]*k[0]); i++) {
		A[i] = (double)(i+1);
	}

	for (i = 0; i < (k[0]*n[0]); i++) {
		B[i] = (double)(-i-1);
	}

	for (i = 0; i < (m[0]*n[0]); i++) {
		C[i] = 0.0;
	}

	//Memory objects for kernel parameters
	cl_mem AMemobj = NULL;
	cl_mem BMemobj = NULL;
	cl_mem CMemobj = NULL;
	cl_mem mMemobj = NULL;
	cl_mem nMemobj = NULL;
	cl_mem kMemobj = NULL;
	cl_mem alphaMemobj = NULL;
	cl_mem betaMemobj = NULL;

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
	char fileName[] = "./dgemm.cl";
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
	AMemobj = clCreateBuffer(context, CL_MEM_READ_ONLY, m[0]*k[0] * sizeof(double), A, &ret);
	BMemobj = clCreateBuffer(context, CL_MEM_READ_ONLY, k[0]*n[0] * sizeof(double), B, &ret);
	CMemobj = clCreateBuffer(context, CL_MEM_READ_WRITE, m[0]*n[0] * sizeof(double), C, &ret);
	mMemobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 1 * sizeof(int), m, &ret);
	nMemobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 1 * sizeof(int), n, &ret);
	kMemobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 1 * sizeof(int), k, &ret);
	alphaMemobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 1 * sizeof(double), alpha, &ret);
	betaMemobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 1 * sizeof(double), beta, &ret);

	//Create kernel program from source
	program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource,(const size_t *)&sourceSize, &ret);

	//Write data to the buffer
	ret = clEnqueueWriteBuffer(commandQueue, AMemobj, CL_TRUE, 0, sizeof(double) * m[0]*k[0], A, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, BMemobj, CL_TRUE, 0, sizeof(double) * k[0]*n[0], B, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, CMemobj, CL_TRUE, 0, sizeof(double) * m[0]*n[0], C, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, mMemobj, CL_TRUE, 0, 1 * sizeof(int), m, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, nMemobj, CL_TRUE, 0, 1 * sizeof(int), n, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, kMemobj, CL_TRUE, 0, 1 * sizeof(int), k, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, alphaMemobj, CL_TRUE, 0, 1 * sizeof(double), alpha, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(commandQueue, betaMemobj, CL_TRUE, 0, 1 * sizeof(double), beta, 0, NULL, NULL);

	//Build the kernel program
	ret = clBuildProgram(program, 1, &deviceId, "-I ./", NULL, NULL);

	//Create a opencl kernel
	kernel = clCreateKernel(program, "dgemm", &ret);	

	//Pass arguments to kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&AMemobj);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&BMemobj);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&CMemobj);
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&mMemobj);
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&nMemobj);
	ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&kMemobj);
	ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&alphaMemobj);
	ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&betaMemobj);

	// Execute OpenCL kernel in data parallel
    const int TS = 32;
	const size_t local[2] = { TS, TS };
	const size_t global[2] = { (int)(pow(2,ceil(log(m[0])/log(2)))), (int)(pow(2,ceil(log(n[0])/log(2)))) };

    //Execute opencl kernel
    ret = clEnqueueNDRangeKernel( commandQueue, kernel, 2, NULL, global, local, 0, 0, 0 );

	//copy the result back
	ret = clEnqueueReadBuffer(commandQueue, CMemobj, CL_TRUE, 0, m[0]*n[0]*sizeof(double), C, 0, NULL, NULL);

	//Before program termination
	ret = clFlush(commandQueue);
	ret = clFinish(commandQueue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(AMemobj);
	ret = clReleaseMemObject(BMemobj);
	ret = clReleaseMemObject(CMemobj);
	ret = clReleaseMemObject(mMemobj);
	ret = clReleaseMemObject(nMemobj);
	ret = clReleaseMemObject(kMemobj);
	ret = clReleaseMemObject(alphaMemobj);
	ret = clReleaseMemObject(betaMemobj);
	ret = clReleaseCommandQueue(commandQueue);
	ret = clReleaseContext(context);
	 
	free(kernelSource);
	free(A);
	free(B);
	free(C);
	free(m);
	free(n);
	free(k);
	free(alpha);
	free(beta);

	//Printing timing result
	end = clock();
	time += (double)(end - begin) / CLOCKS_PER_SEC;

	printf("%d\n", time);

	return 0;
}