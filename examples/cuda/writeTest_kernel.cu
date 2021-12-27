extern "C" {

	__global__ void	writeTest(int* array){
		for(int i = 0; i < 8; i++){
			array[i] *= 2;
		}
	}

}
