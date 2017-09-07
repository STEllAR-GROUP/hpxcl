extern "C" { __global__ void multiply_step(size_t* size, double* in, double* out,double* factor) {

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < size[0];
			i += gridDim.x * blockDim.x) {
		out[i] = in[i] * factor[0];
	}
}
}

extern "C" { __global__ void add_step(size_t* size, double* in, double* in2, double* out) {

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < size[0];
			i += gridDim.x * blockDim.x) {
		out[i] = in[i] + in2[i];
	}
}
}

extern "C" { __global__ void triad_step(size_t* size, double* in, double* in2, double* out,double* factor) {

	for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < size[0];
			i += gridDim.x * blockDim.x) {
		out[i] = in[i] + in2[i] * factor[0];
	}
}
}
