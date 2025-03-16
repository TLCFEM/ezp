FROM opensuse/tumbleweed AS build

RUN zypper in -y gcc-c++ cmake libscalapack2_2_2_0-gnu-mpich-hpc-devel-static

COPY . /ezp

WORKDIR /ezp

RUN mkdir build && cd build && \
    cmake \
    -DEZP_STANDALONE=ON \
    -DEZP_USE_SYSTEM_LIBS=ON \
    -DMPI_HOME=/usr/lib/hpc/gnu14/mpi/mpich/4.3.0/ \
    -DCMAKE_PREFIX_PATH="/usr/lib/hpc/gnu14/mpich/scalapack/2.2.0/lib64/;/usr/lib/hpc/gnu14/openblas/0.3.29/lib64/" \
    -DCMAKE_INSTALL_PREFIX=/ezp-install \
    .. && cmake --build . --target install

FROM opensuse/tumbleweed AS runtime

RUN zypper in -y libscalapack2_2_2_0-gnu-mpich-hpc

ENV LD_LIBRARY_PATH=/usr/lib/hpc/gnu14/mpich/scalapack/2.2.0/lib64/:/usr/lib/hpc/gnu14/openblas/0.3.29/lib64/

COPY --from=build /ezp-install/* /bin
