#if !defined(KERNEL_CUH)
#define KERNEL_CUH
void cuda_malloc(void **devPtr, size_t size);
void cuda_test(long int* a);
int get_devices();
void get_device_info();
long int gpu_num_of_hits(int blocks,int threads,int trials_per_thread);
#endif //KERNEL_CUH
