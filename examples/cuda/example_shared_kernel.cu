extern "C" { 
	__global__ void dynamicReverse(int* d, int* sizebuffer){
		const int n = sizebuffer[0];

		extern __shared__ int s[];

		int t = threadIdx.x;
		int tr = n-t-1;
		
		s[t] = d[t];
		
		__syncthreads();

		d[t] = s[tr];
	}
}
