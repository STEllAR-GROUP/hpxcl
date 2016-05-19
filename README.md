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
- HPX >= 0.9

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
-DCUDA_TOOLKIT_ROOT_DIR=/opt/packages/cuda-7.0/ 
```

OpenCl
==

Prerequisites:

- OpenCl >= 1.1

Building:
```
mkdir build && cd build
cmake \
-DHPX_ROOT=/home/diehl/opt/hpx/ \
-DHPXCL_WITH_OPENCL=ON 
```
Options
==

- Build CUDA support: -HPXCL_WITH_CUDA (Default=Off)
- Build OpenCL support: -HPCL_WITH_OPENCL (Default=Off)
- Build examples: -DHPXCL_BUILD_EXAMPLES (Default=Off)
- Build benchmark: -DHPXCL_BUILD_BENCHMARK (Default=Off)
- Build documentation: -DHPX_BUILD_DOCUMENTATION (Defaut=Off)
