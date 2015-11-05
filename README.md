hpxcl
=====

This repository contains components that will support percolation via OpenCL and CUDA

Build
====

The `CircleCI <https://circleci.com/gh/STEllAR-GROUP/hpxcl>`_ contiguous
integration service tracks the current build status for the master branch:
|circleci_status|

.. |circleci_status| image:: https://circleci.com/gh/STEllAR-GROUP/hpxcl/tree/master.svg?style=svg
     :target: https://circleci.com/gh/STEllAR-GROUP/hpxcl/tree/master
     :alt: HPXCL master branch build status

CUDA
===

Prerequisites:

- CUDA SDK > 7.5

Building:
```
mkdir build && cd build
cmake \
-DHPX_ROOT=/home/diehl/opt/hpx/ \
-DHPXCL_WITH_CUDA=ON \
-DCUDA_TOOLKIT_ROOT_DIR=/opt/packages/cuda-7.0/ \
-DLIBNVRTC_LIBRARY_DIR=/opt/packages/cuda-7.0/lib64/ .. 
```
