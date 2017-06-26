hpxcl
====

This repository contains components that will support percolation via OpenCL and CUDA

The [documentation](http://stellar-group.github.io/hpxcl/docs/html/index.html) of the nightly build is available.

In any case, if you happen to run into problems we very much encourage and appreciate
any [issue](http://github.com/STEllAR-GROUP/hpxcl/issues) reports through the issue tracker for this Github project.

The [CircleCI](https://circleci.com/gh/STEllAR-GROUP/hpxcl) contiguous
integration service tracks the current build status for the master branch:
![HPXCL master branch build status](https://circleci.com/gh/STEllAR-GROUP/hpxcl/tree/master.svg?style=svg "")

Build
===

- CMake >= 3.0
- GCC >= 4.9 
- HPX (Please use the current version in the master branch)

CUDA
==

Prerequisites:

- CUDA SDK >= 7.0

Building:
```
mkdir build && cd build
cmake \
-DHPX_ROOT=/home/ubuntu/opt/hpx/ \
-DHPXCL_WITH_CUDA=ON \
-DCUDA_TOOLKIT_ROOT_DIR=/opt/packages/cuda-7.0/	\
..
```

OpenCl
==

Prerequisties:

- OpenCl >= 1.1

Building:
```
mkdir build && cd build
cmake	\
-DHPX_ROOT=/home/ubuntu/opt/hpx	\
-DHPXCL_WITH_OPENCL=ON \
..
```


Options
==

- Build CUDA support: -HPXCL_WITH_CUDA (Default=Off)
- Build examples: -DHPXCL_BUILD_EXAMPLES (Default=On)
- Build benchmark: -DHPXCL_WITH_BENCHMARK (Default=Off)
- Build documentation: -DHPX_WITH_DOCUMENTATION (Defaut=Off)
- Build the naive CUDA benchmarks: -DHPXCL_WITH_NAIVE_CUDA_BENCHMARK (DEFAULT=Off)
