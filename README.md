hpxcl
====

This repository contains components that will support percolation via OpenCL and CUDA

In any case, if you happen to run into problems we very much encourage and appreciate
any [issue](http://github.com/STEllAR-GROUP/hpxcl/issues) reports through the issue tracker for this Github project.

Build
===

The [CircleCI](https://circleci.com/gh/STEllAR-GROUP/hpxcl) contiguous
integration service tracks the current build status for the master branch:
![HPXCL master branch build status](https://circleci.com/gh/STEllAR-GROUP/hpxcl/tree/master.svg?style=svg "")

CUDA
==

Prerequisites:

- CUDA SDK >= 7.0

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
- Build examples: -DHPXCL_BUILD_EXAMPLES
- Build benchmark: -DHPXCL_BUILD_BENCHMARK=ON  
- Build documentation: -DHPX_BUILD_DOCUMENTATION=ON
