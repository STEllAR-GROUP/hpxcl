#if !defined(KERNEL_CUH)
#define KERNEL_CUH
void cuda_malloc(void **devPtr, size_t size);
void cuda_test(long int* a);
int get_devices();
void get_device_info();
float pi(int nthreads,int nblocks);
#endif //KERNEL_CUH
