hpxcl
====

This repository contains components that will support percolation via OpenCL and CUDA

Build
===

The [CircleCI](https://circleci.com/gh/STEllAR-GROUP/hpxcl) contiguous
integration service tracks the current build status for the master branch:
![HPXCL master branch build status](https://circleci.com/gh/STEllAR-GROUP/hpxcl/tree/master.svg?style=svg "")

CUDA
==

Prerequisites:

- CUDA SDK >= 7.0
- HPX >= 0.9

Building:
```
mkdir build && cd build
cmake \
-DHPX_ROOT=/home/diehl/opt/hpx/ \
-DHPXCL_WITH_CUDA=ON \
-DCUDA_TOOLKIT_ROOT_DIR=/opt/packages/cuda-7.0/ \
-DLIBNVRTC_LIBRARY_DIR=/opt/packages/cuda-7.0/lib64/ .. 
```

- Build CUDA support: -HPXCL_WITH_CUDA=ON
- Build OpenCL support: -HPCL_WITH_OPENCL=ON
- Build examples for CUDA:  -DHPXCL_WITH_CUDA_EXAMPLES=ON
- Build benchmark for CUDA: -DHPXCL_WITH_CUDA_BENCHMARK=ON
- Build benchmark for OPENCL: -DHPXCL_WITH_OPENCL_BENCHMARK=ON
