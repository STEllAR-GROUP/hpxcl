
extern "C" __global__ void sum(unsigned int* array, unsigned int* count,
		unsigned int* n) {
	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n[0];
			i += gridDim.x * blockDim.x) {
		atomicAdd(&(count[0]), array[i]);
	}
}
