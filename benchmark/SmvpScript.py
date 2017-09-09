# Copyright (c)       2017 Madhavan Seshadri
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

##############################################################################################################
#This Script uses varying k values to benchmark the smvp
##############################################################################################################

import os
import csv
import subprocess
import sys

#for headless display in python
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

if(len(sys.argv) != 3):
    print("Usage #node_name #retry_attempts")
	sys.exit()
	
############Parameters for varying k#######
start = 10000
end = 100000
step = 10000

#Load modules
os.system("module load cmake/3.7.2 gcc/4.9.4 cuda/8.0.61")

os.chdir("benchmark/cuda/smvp/")
os.system("rm *.dat")

os.system("cmake -DHPX_ROOT=~/packagaes/hpx-4.9/ -DHPXCL_WITH_CUDA=ON -DHPXCL_WITH_OPENCL=ON -DHPXCL_WITH_BENCHMARK=ON -DHPXCL_WITH_NAIVE_OPENCL_BENCHMARK=ON -DHPXCL_WITH_NAIVE_CUDA_BENCHMARK=ON -DOPENCL_ROOT=/usr/local/cuda-8.0/ -DHPXCL_CUDA_WITH_STREAM=ON ~/hpxcl/")
os.system("make")

######################################Profiling for the CUDA part #############################################
print ('Profiling Naive CUDA SMVP......\n')
#profiling SMVP HPX code
for i in range(start,end,step):
    try_again = int(sys.argv[2])
    for j in range(try_again,0,-1):
        try:
            subprocess.call("srun -p "+ str(sys.argv[1]) + " -N 1 ./smvpHPXCL 10240 " + str(i) + " >> smvpHPX_cuda.dat",shell=True)
            break
        except OSError:
            print ('trying again......\n')
            
print ('Profiling HPXCL CUDA SMVP......\n')
#profiling SMVP Cuda code
for i in range(start,end,step):
    try_again = int(sys.argv[2])
    for j in range(try_again,0,-1):
        try:
            subprocess.call("srun -p "+ str(sys.argv[1]) + " -N 1 ./smvpCUDA 10240 " + str(i) + " >> smvpCUDA.dat",shell=True)
            break
        except OSError:
            print ('trying again......\n')

smvpCudaX = []
smvpCudaY = []
with open('smvpCUDA.dat', 'rb') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    xValue = start
    for row in reader:
        smvpCudaX.append(int(xValue))
        smvpCudaY.append(float(row[0]))
        xValue += step

plt.plot(smvpCudaX, smvpCudaY,marker='.', linestyle=':', color='b', label='Naive CUDA')

smvpHpxX = []
smvpHpxY = []
with open('smvpHPX_cuda.dat', 'rb') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    xValue = start
    for row in reader:
        smvpHpxX.append(int(xValue))
        smvpHpxY.append(float(row[0]))
        xValue += step

plt.plot(smvpHpxX, smvpHpxY,marker='.', linestyle='-', color='r', label='HPXCL CUDA')

######################################Profiling for the OpenCL part #############################################
os.chdir("../../opencl/smvp/")
os.system("rm *.dat")

print ('Profiling HPXCL OpenCL SMVP......\n')
#profiling smvp HPX code
for i in range(start,end,step):
    try_again = int(sys.argv[2])
    for j in range(try_again,0,-1):
        try:
            subprocess.call("srun -p "+ str(sys.argv[1]) + " -N 1 ./smvpHPX 10240 " + str(i) + " >> smvpHPX_opencl.dat",shell=True)
            break
        except OSError:
            print ('trying again......\n')

print ('Profiling Naive OpenCL SMVP......\n')
#profiling smvp Cuda code
for i in range(start,end,step):
    try_again = int(sys.argv[2])
    for j in range(try_again,0,-1):
        try:
            subprocess.call("srun -p "+ str(sys.argv[1]) + " -N 1 ./smvp_opencl 10240 " + str(i) + " >> smvpCL.dat",shell=True)
            break
        except OSError:
            print ('trying again......\n')

smvpOpenclX = []
smvpOpenclY = []
with open('smvpCL.dat', 'rb') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    xValue = start
    for row in reader:
        smvpOpenclX.append(int(xValue))
        smvpOpenclY.append(float(row[0]))
        xValue += step

plt.plot(smvpOpenclX, smvpOpenclY,marker='o', linestyle='-.', color='g', label='Naive OpenCL')

smvpHpxOpenclX = []
smvpHpxOpenclY = []
with open('smvpHPX_opencl.dat', 'rb') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    xValue = start
    for row in reader:
        smvpHpxOpenclX.append(int(xValue))
        smvpHpxOpenclY.append(float(row[0]))
        xValue += step

plt.plot(smvpHpxOpenclX, smvpHpxOpenclY,marker='o', linestyle='-', color='k', label='HPXCL OpenCL')

plt.xlabel('n')
plt.ylabel('Time in milliseconds')
plt.title('n vs. Time SMVP Benchmark')
plt.legend()
plt.grid()
os.chdir("../../")
plt.savefig('smvp.png')