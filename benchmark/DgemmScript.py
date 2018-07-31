# Copyright (c)       2017 Madhavan Seshadri
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

##############################################################################################################
#This Script uses varying k values to benchmark the dgemm 
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

############Parameters for varying m#######
m_value = 10240

############Parameters for varying n#######	
n_value = 10240

############Parameters for varying k#######
start = 1000
end = 11000
step = 1000

#Load modules
os.system("module load cmake/3.7.2 gcc/4.9.4 cuda/8.0.61")

os.chdir("benchmark/cuda/dgemm/")
os.system("rm *.dat")

os.system("cmake -DHPX_ROOT=~/packagaes/hpx-4.9/ -DHPXCL_WITH_CUDA=ON -DHPXCL_WITH_OPENCL=ON -DHPXCL_WITH_BENCHMARK=ON -DHPXCL_WITH_NAIVE_OPENCL_BENCHMARK=ON -DHPXCL_WITH_NAIVE_CUDA_BENCHMARK=ON -DOPENCL_ROOT=/usr/local/cuda-8.0/ -DHPXCL_CUDA_WITH_STREAM=ON ~/hpxcl/")
os.system("make")

######################################Profiling for the CUDA part #############################################
print ('Profiling Naive CUDA DGEMM......\n')
#profiling Dgemm HPX code
for i in range(start,end,step):
    try_again = int(sys.argv[2])
    for j in range(try_again,0,-1):
        try:
            subprocess.call("srun -p "+ str(sys.argv[1]) + " -N 1 ./dgemmHPXCL "+str(m_value)+" "+str(n_value)+" " + str(i) + " >> dgemmHPX_cuda.dat",shell=True)
            break
        except OSError:
            print ('trying again......\n')
            
print ('Profiling HPXCL CUDA DGEMM......\n')
#profiling Dgemm Cuda code
for i in range(start,end,step):
    try_again = int(sys.argv[2])
    for j in range(try_again,0,-1):
        try:
            subprocess.call("srun -p "+ str(sys.argv[1]) + " -N 1 ./dgemmCUDA "+str(m_value)+" "+str(n_value)+" " + str(i) + " >> dgemmCUDA.dat",shell=True)
            break
        except OSError:
            print ('trying again......\n')

dgemmCudaX = []
dgemmCudaY = []
with open('dgemmCUDA.dat', 'rb') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    xValue = start
    for row in reader:
        dgemmCudaX.append(int(xValue))
        dgemmCudaY.append(float(row[0]))
        xValue += step

plt.plot(dgemmCudaX, dgemmCudaY,marker='.', linestyle=':', color='b', label='Naive CUDA')

dgemmHpxX = []
dgemmHpxY = []
with open('dgemmHPX_cuda.dat', 'rb') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    xValue = start
    for row in reader:
        dgemmHpxX.append(int(xValue))
        dgemmHpxY.append(float(row[0]))
        xValue += step

plt.plot(dgemmHpxX, dgemmHpxY,marker='.', linestyle='-', color='r', label='HPXCL CUDA')
"""
######################################Profiling for the OpenCL part #############################################
os.chdir("../../opencl/dgemm/")
os.system("rm *.dat")

print ('Profiling HPXCL OpenCL DGEMM......\n')
#profiling Dgemm HPX code
for i in range(start,end,step):
    try_again = int(sys.argv[2])
    for j in range(try_again,0,-1):
        try:
            subprocess.call("srun -p "+ str(sys.argv[1]) + " -N 1 ./dgemmHPX "+str(m_value)+" "+str(n_value)+" " + str(i) + " >> dgemmHPX_opencl.dat",shell=True)
            break
        except OSError:
            print ('trying again......\n')

print ('Profiling Naive OpenCL DGEMM......\n')
#profiling Dgemm Cuda code
for i in range(start,end,step):
    try_again = int(sys.argv[2])
    for j in range(try_again,0,-1):
        try:
            subprocess.call("srun -p "+ str(sys.argv[1]) + " -N 1 ./dgemm_opencl "+str(m_value)+" "+str(n_value)+" " + str(i) + " >> dgemmCL.dat",shell=True)
            break
        except OSError:
            print ('trying again......\n')

dgemmOpenclX = []
dgemmOpenclY = []
with open('dgemmCL.dat', 'rb') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    xValue = start
    for row in reader:
        dgemmOpenclX.append(int(xValue))
        dgemmOpenclY.append(float(row[0]))
        xValue += step

plt.plot(dgemmOpenclX, dgemmOpenclY,marker='o', linestyle='-.', color='g', label='Naive OpenCL')

dgemmHpxOpenclX = []
dgemmHpxOpenclY = []
with open('dgemmHPX_opencl.dat', 'rb') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    xValue = start
    for row in reader:
        dgemmHpxOpenclX.append(int(xValue))
        dgemmHpxOpenclY.append(float(row[0]))
        xValue += step

plt.plot(dgemmHpxOpenclX, dgemmHpxOpenclY,marker='o', linestyle='-', color='k', label='HPXCL OpenCL')
"""
plt.xlabel('k')
plt.ylabel('Time in milliseconds')
plt.title('k vs. Time DGEMM Benchmark')
plt.legend()
plt.grid()
os.chdir("../../")
plt.savefig('dgemm.png')