hpxcl
====

This repository contains components that will support percolation via OpenCL and CUDA

Build
===

The [CircleCI](https://circleci.com/gh/STEllAR-GROUP/hpxcl)_ contiguous
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
