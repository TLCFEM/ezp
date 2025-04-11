FROM opensuse/tumbleweed AS build

RUN zypper in -y gcc-c++ gcc-fortran cmake libscalapack2-openmpi5-devel-static openblas_openmp-devel-static

COPY . /ezp

WORKDIR /ezp

RUN mkdir build && cd build && \
    cmake \
    -DEZP_STANDALONE=ON \
    -DEZP_USE_SYSTEM_LIBS=ON \
    -DMPI_HOME=/usr/lib64/mpi/gcc/openmpi5/ \
    -DCMAKE_PREFIX_PATH="/usr/lib64/mpi/gcc/openmpi5/lib64/;/usr/lib64/openblas-openmp/" \
    -DCMAKE_INSTALL_PREFIX=/ezp-install \
    .. && cmake --build . --target install

FROM opensuse/tumbleweed AS runtime

RUN zypper in -y libscalapack2-openmpi5 libopenblas_openmp0 openmpi5

ENV LD_LIBRARY_PATH=/usr/lib64/mpi/gcc/openmpi5/lib64/

COPY --from=build /ezp-install/* /bin
