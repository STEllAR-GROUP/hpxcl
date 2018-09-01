// Copyright (c)       2017 Madhavan Seshadri
//                     2018 Patrick Diehl
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
extern "C" { __global__ void kernel(char *out, int *width, int *height, int *yStart, int* n){
	unsigned int xDim = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yDim = blockIdx.y * blockDim.y + threadIdx.y;

	//index of the output array, multiplied by 3 for R,G,B values
	int arrayIndex = 3 * (*width) * yDim + xDim*3;
    
	float xPoint = ((float) (xDim)/(*width)) * 3.25f - 2.0f;
	float yPoint = ((float) (yDim+*yStart)/(*height)) * 2.5f - 1.25f; 

	//for calculation of complex number
	float x = 0.0;
	float y = 0.0;

	int iterationCount = 0;
	int numIterations = 256;
	//terminating condition x^2+y^2 < 4 or iterations >= numIterations
	while(y*y+x*x<=4 && iterationCount<(numIterations)){
		float xTemp = x*x-y*y + xPoint;
		y = 2*x*y + yPoint;
		x = xTemp;
		iterationCount++;
	}
    
    if (arrayIndex < *n)
    { 
	if(iterationCount == (numIterations)){
		out[arrayIndex] = iterationCount;
		out[arrayIndex+1]=1;
		out[arrayIndex+2]=iterationCount;
	}else{
		out[arrayIndex] = 0;
		out[arrayIndex+1]=iterationCount;
		out[arrayIndex+2]=0;
	}
   }
 }
};
