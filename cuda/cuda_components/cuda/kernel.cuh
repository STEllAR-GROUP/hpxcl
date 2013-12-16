#if !defined(KERNEL_CUH)
#define KERNEL_CUH

void cuda_test(long int* a);

int get_devices();

void get_device_info();

void set_device(int dev);

float pi(int nthreads,int nblocks);

#endif //KERNEL_CUH
