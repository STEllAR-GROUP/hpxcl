import os
import csv

#for headless display in python
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

#Load modules
os.system("module load cmake/3.7.2 gcc/4.9.4")
os.chdir("benchmark/cuda/stencil")

os.system("rm *.dat")

os.system("cmake -DHPX_ROOT=~/packagaes/hpx-4.9/ -DHPXCL_WITH_CUDA=ON -DHPXCL_WITH_BENCHMARK=ON -DHPXCL_WITH_NAIVE_CUDA_BENCHMARK=ON -DHPXCL_CUDA_WITH_STREAM=OFF -DOPENCL_ROOT=/usr/local/cuda-8.0/ ~/hpxcl/")
os.system("make")

#profiling Stencil HPX code
for i in range(1,8):
	os.system("srun -p tycho -N 1 ./StencilHPX " + str(i) + " >> StencilHPX.dat")

#profiling Stencil Cuda code
for i in range(1,8):
	os.system("srun -p tycho -N 1 ./StencilCuda " + str(i) + " >> StencilCuda.dat")

stencilCudaX = []
stencilCudaX = []
with open('StencilCuda.dat', 'rb') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    for row in reader:
        stencilCudaX.append(int(row[0]))
        stencilCudaY.append(float(row[2]))

plt.plot(stencilCudaX, stencilCudaX, label='Naive CUDA')

stencilHpxX = []
stencilHpxY = []
with open('StencilHPX.dat', 'rb') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    for row in reader:
        stencilHpxX.append(int(row[0]))
        stencilHpxY.append(float(row[2]))

plt.plot(stencilHpxX, stencilHpxY,marker='o', linestyle='--', color='r', label='HPXCL CUDA')

os.system("cmake -DHPX_ROOT=~/packagaes/hpx-4.9/ -DHPXCL_WITH_CUDA=ON -DHPXCL_WITH_BENCHMARK=ON -DHPXCL_WITH_NAIVE_CUDA_BENCHMARK=ON -DHPXCL_CUDA_WITH_STREAM=ON -DOPENCL_ROOT=/usr/local/cuda-8.0/ ~/hpxcl/")
os.system("make")


#profiling Partition HPX code
for i in range(1,8):
	os.system("srun -p tycho -N 1 ./StencilHPX " + str(i) + " >> StencilStreamHPX.dat")

stencilStreamHpxX = []
stencilStreamHpxY = []
with open('StencilStreamHPX.dat', 'rb') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    for row in reader:
        stencilStreamHpxX.append(int(row[0]))
        stencilStreamHpxY.append(float(row[2]))

plt.plot(stencilStreamHpxX, stencilStreamHpxY,marker='o', linestyle='--', color='b', label='HPXCL CUDA WITH STREAM')

plt.xlabel('n')
plt.ylabel('Time in milliseconds')
plt.title('n vs. Time Multiple Stencil Benchmark')
plt.legend()
plt.savefig('Stencil.png')