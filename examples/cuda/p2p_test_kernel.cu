extern "C" { 
	__global__ void multiply_1(int* input, int* output, int* sizebuffer) {
		int size = sizebuffer[0];

		for(int i = 0; i < size/blockDim.x; i++){
			output[threadIdx.x * (size/blockDim.x) + i] = input[threadIdx.x * (size/blockDim.x) + i]*2;
		}
	}

	__global__ void multiply_2(int* input, int* output, int* sizebuffer) {
		int size = sizebuffer[0];

		for(int i = 0; i < size/blockDim.x; i++){
			output[threadIdx.x * (size/blockDim.x) + i] = input[threadIdx.x * (size/blockDim.x) + i]*4;
		}
	}
}
