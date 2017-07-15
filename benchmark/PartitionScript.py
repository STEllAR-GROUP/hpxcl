import os
import csv
import matplotlib.pyplot as plt

#Load modules
os.system("module load cmake/3.7.2 gcc/4.9.4")
os.chdir("benchmark/cuda/stencil")

os.system("cmake -DHPX_ROOT=~/packagaes/hpx-4.9/ -DHPXCL_WITH_CUDA=ON -DHPXCL_WITH_BENCHMARK=ON -DHPXCL_WITH_NAIVE_CUDA_BENCHMARK=ON -DHPXCL_CUDA_WITH_STREAM=OFF -DOPENCL_ROOT=/usr/local/cuda-8.0/ ~/hpxcl/")
os.system("make")

#profiling Partition Cuda code
for i in range(1,8):
	os.system("srun -p tycho -N 1 ./PartitionCuda " + str(i) + " >> PartitionCuda.dat")

#profiling Partition HPX code
for i in range(1,8):
	os.system("srun -p tycho -N 1 ./PartitionHPX " + str(i) + " >> PartitionHPX.dat")

#profiling Stencil HPX code
for i in range(1,8):
	os.system("srun -p tycho -N 1 ./StencilHPX " + str(i) + " >> StencilHPX.dat")

#profiling Stencil HPX code
for i in range(1,8):
	os.system("srun -p tycho -N 1 ./StencilCuda " + str(i) + " >> StencilCuda.dat")

partitionCudaX = []
partitionCudaY = []
with open('PartitionCuda.dat', 'rb') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    for row in reader:
        partitionCudaX.append(int(row[0]))
        partitionCudaY.append(int(row[2]))

plt.plot(partitionCudaX, partitionCudaY, label='Naive CUDA')

partitionHpxX = []
partitionHpxY = []
with open('PartitionHPX.dat', 'rb') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    for row in reader:
        partitionHpxX.append(int(row[0]))
        partitionHpxY.append(int(row[2]))

plt.plot(partitionCudaX, partitionCudaY,marker='o', linestyle='--', color='r', label='HPXCL CUDA')

os.system("cmake -DHPX_ROOT=~/packagaes/hpx-4.9/ -DHPXCL_WITH_CUDA=ON -DHPXCL_WITH_BENCHMARK=ON -DHPXCL_WITH_NAIVE_CUDA_BENCHMARK=ON -DHPXCL_CUDA_WITH_STREAM=ON -DOPENCL_ROOT=/usr/local/cuda-8.0/ ~/hpxcl/")
os.system("make")

#profiling Partition HPX code
for i in range(1,8):
	os.system("srun -p tycho -N 1 ./PartitionHPX " + str(i) + " >> PartitionStreamHPX.dat")

partitionStreamHpxX = []
partitionStreamHpxY = []
with open('PartitionStreamHPX.dat', 'rb') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    for row in reader:
        partitionStreamHpxX.append(int(row[0]))
        partitionStreamHpxY.append(int(row[2]))

plt.plot(partitionCudaX, partitionCudaY,marker='o', linestyle='--', color='b', label='HPXCL CUDA WITH STREAM')

plt.xlabel('n')
plt.ylabel('Time in milliseconds')
plt.title('n vs. Time Multiple Partition Benchmark')
plt.legend()
plt.savefig('Partition.png')