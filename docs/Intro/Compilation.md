# Compilation

As `ezp` is wrapper library of `ScaLAPACK`, the following dependencies are necessary to compile any executables that use `ezp`.

1. an implementation of `LAPACK` and `BLAS`, such as `OpenBLAS`, `Intel® oneAPI Math Kernel Library (oneMKL)`, `AMD Optimizing CPU Libraries (AOCL)`, etc.
2. an implementation of `ScaLAPACK`, such as the reference implementation, Intel's implementation, NVIDIA's implementation, etc.
3. an implementation of `MPI`, such as `OpenMPI`, `MPICH`, `Intel® MPI` etc.

Before compiling the executables, one must ensure those libraries are available.

## oneAPI

The easiest approach is to use the `Intel® oneAPI` toolkit.
It provides the `Intel® MPI Library` and the `Intel® oneAPI Math Kernel Library (oneMKL)` which contains a complete `ScaLAPACK` and `LAPACK`/`BLAS` implementation.

The following is a sample workflow that runs on `Fedora`.

```bash
echo -e "[oneAPI]\nname=Intel® oneAPI repository\nbaseurl=https://yum.repos.intel.com/oneapi\nenabled=1\ngpgcheck=1\nrepo_gpgcheck=1\ngpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB" > /etc/yum.repos.d/oneAPI.repo

sudo dnf install -y intel-oneapi-mkl-devel intel-oneapi-mpi-devel cmake gcc g++ gfortran git ninja-build

git clone --recurse-submodules --depth 1 https://github.com/TLCFEM/ezp.git

mkdir /ezp/build && /ezp/build

. /opt/intel/oneapi/setvars.sh && cmake -DEZP_TEST=ON .. && cmake --build . --config Release
```

## System Libraries

Using system libraries is possible but most bundled libraries are broken/outdated on various distros.
It's likely the bundled system libraries do not work for various compatibility issues.
To circumvent, one then needs to manually compile all those dependencies before using them.
This is cumbersome.

`openSUSE` has a functioning toolset.
Simply install the library and you are ready to go.

```bash
sudo zypper in -y gcc-c++ gcc-fortran cmake libscalapack2-openmpi5-devel-static openblas_openmp-devel-static
git clone --recurse-submodules --depth 1 https://github.com/TLCFEM/ezp.git
mkdir /ezp/build && /ezp/build
cmake -DEZP_STANDALONE=ON -DEZP_USE_SYSTEM_LIBS=ON -DMPI_HOME=/usr/lib64/mpi/gcc/openmpi5/ -DCMAKE_PREFIX_PATH="/usr/lib64/mpi/gcc/openmpi5/lib64/;/usr/lib64/openblas-openmp/" .. && cmake --build . --config Release
```

`Fedora 41` has an environment that is close to usable.
The following is an example that uses system packages.

```bash
sudo dnf install -y cmake gcc g++ gfortran git ninja-build scalapack-mpich-devel flexiblas-devel
# sadly the scalapack package is broken on fedora 41
# need to provide this link
sudo ln -s /usr/lib64/mpich/lib/libscalapack.so.2.2.0 /usr/lib64/libscalapack.so.2.2.0

git clone --recurse-submodules --depth 1 https://github.com/TLCFEM/ezp.git
mkdir /ezp/build && /ezp/build
cmake -DEZP_TEST=ON -DMPI_HOME=/usr/lib64/mpich/ -DEZP_USE_SYSTEM_LIBS=ON .. && cmake --build . --config Release
```

Other mainstream distros require more fixes.
Still, it is not recommended due to the lack of flexibility.
It may not be possible/feasible to switch to another implementation of any of those libraries.

## Compilation Options

### 64-bit Integer

Use `-DEZP_USE_64BIT_INT=ON` flag in CMake, or define the macro `EZP_INT64`.

### Name Mangling

Use `-DEZP_ADD_UNDERSCORE=ON` flag in CMake, or define the macro `EZP_UNDERSCORE`.

### Standalone Solvers

Use `-DEZP_STANDALONE=ON` flag in CMake to compile standalone solver.

### Enable OpenMP

Use `-DEZP_ENABLE_OPENMP=ON` flag in CMake to enable the `-fopenmp` compiling flag.
