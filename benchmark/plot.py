import os
import csv

#for headless display in python
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

stencilCudaX = []
stencilCudaY = []
with open('StencilCuda.dat', 'rb') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    for row in reader:
        stencilCudaX.append(int(row[0]))
        stencilCudaY.append(float(row[2]))

plt.plot(stencilCudaX, stencilCudaY, label='Naive CUDA')

stencilHpxX = []
stencilHpxY = []
with open('StencilHPX.dat', 'rb') as f:
    reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
    for row in reader:
        stencilHpxX.append(int(row[0]))
        stencilHpxY.append(float(row[2]))

plt.plot(stencilHpxX, stencilHpxY,marker='o', linestyle='--', color='r', label='HPXCL CUDA')

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