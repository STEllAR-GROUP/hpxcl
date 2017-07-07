# Copyright (c) 2015 Martin Stumpf
#		2016 Patrick Diehl
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

FROM stellargroup/hpx:dev
RUN apt-get autoremove -y && apt-get clean && apt-get update && apt-get install -y xsltproc      \
        doxygen								\
	libpng-dev								

#Installation of CUDA 7.0
RUN wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run && \
    chmod +x cuda_8.0.61_375.26_linux-run && \
    mkdir nvidia_installers && \
    ./cuda_8.0.61_375.26_linux-run -extract=`pwd`/nvidia_installers && \
    cd nvidia_installers && \
    ./cuda-linux64-rel-8.0.61-21551265.run -noprompt && \
    cd .. && \
    rm -r nvidia_installers

ENV LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64
ENV PATH=$PATH:/usr/local/cuda-8.0/bin
ENV CUDA_LIB_PATH=/usr/local/cuda-8.0/lib64/stubs/


