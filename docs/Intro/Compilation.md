# Compilation

`ezp` is a header-only wrapper around `ScaLAPACK`.
Compiling an executable that uses `ezp` therefore requires the following dependencies.

1. an implementation of `LAPACK` and `BLAS`, such as `OpenBLAS`, Intel® oneMKL, or `AOCL`;
2. an implementation of `ScaLAPACK`, such as the reference implementation or the one bundled with `Intel® oneMKL`;
3. an implementation of `MPI`, such as `OpenMPI`, `MPICH`, or Intel® MPI.

Ensure these libraries are available before configuring the build.
[C++20](https://en.cppreference.com/w/cpp/20) or later is required.

## oneAPI

The easiest approach is to use the Intel® oneAPI toolkit.
It provides the Intel® MPI Library and Intel® oneMKL, which bundles a complete `ScaLAPACK` and `LAPACK`/`BLAS` implementation.

The following workflow runs on Fedora.
It installs the required packages, checks out the repository, and builds with the oneAPI environment.

```bash
echo -e "[oneAPI]\nname=Intel® oneAPI repository\nbaseurl=https://yum.repos.intel.com/oneapi\nenabled=1\ngpgcheck=1\nrepo_gpgcheck=1\ngpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB" > /etc/yum.repos.d/oneAPI.repo

sudo dnf install -y intel-oneapi-mkl-devel intel-oneapi-mpi-devel cmake gcc g++ gfortran git ninja-build

git clone --recurse-submodules --depth 1 https://github.com/TLCFEM/ezp.git

mkdir /ezp/build && cd /ezp/build

. /opt/intel/oneapi/setvars.sh && cmake -DEZP_TEST=ON .. && cmake --build . --config Release
```

The build links against `MKL::MKL_SCALAPACK` and defines the `EZP_MKL` macro automatically.

## System Libraries

System libraries are usable, but many distributions ship broken or outdated versions.
The bundled system libraries often fail due to compatibility issues.
In such cases one must compile all dependencies manually, which is cumbersome.
`openSUSE` provides a toolset that works out of the box.

```bash
sudo zypper in -y gcc-c++ gcc-fortran cmake libscalapack2-openmpi5-devel-static openblas_openmp-devel-static
git clone --recurse-submodules --depth 1 https://github.com/TLCFEM/ezp.git
mkdir /ezp/build && cd /ezp/build
cmake -DEZP_STANDALONE=ON -DEZP_USE_SYSTEM_LIBS=ON -DMPI_HOME=/usr/lib64/mpi/gcc/openmpi5/ -DCMAKE_PREFIX_PATH="/usr/lib64/mpi/gcc/openmpi5/lib64/;/usr/lib64/openblas-openmp/" .. && cmake --build . --config Release
```

Fedora 41 provides an environment close to usable.
The `scalapack` package is broken there, so a manual symlink is required.

```bash
sudo dnf install -y cmake gcc g++ gfortran git ninja-build scalapack-mpich-devel flexiblas-devel
sudo ln -s /usr/lib64/mpich/lib/libscalapack.so.2.2.0 /usr/lib64/libscalapack.so.2.2.0

git clone --recurse-submodules --depth 1 https://github.com/TLCFEM/ezp.git
mkdir /ezp/build && cd /ezp/build
cmake -DEZP_TEST=ON -DMPI_HOME=/usr/lib64/mpich/ -DEZP_USE_SYSTEM_LIBS=ON .. && cmake --build . --config Release
```

Other mainstream distributions require more fixes.
With `EZP_USE_SYSTEM_LIBS`, the build looks for a system `scalapack` library, then for optional `flexiblas` and `openblas` libraries.
Switching among implementations of the underlying libraries is not recommended, because it lacks flexibility.

## Compilation Options

### 64-bit Integer

Use `-DEZP_USE_64BIT_INT=ON` in CMake, or define the macro `EZP_INT64`.
This switches the `MKL_INTERFACE` from `lp64` to `ilp64`.

### Name Mangling

Use `-DEZP_ADD_UNDERSCORE=ON` in CMake, or define the macro `EZP_UNDERSCORE`.
This controls whether Fortran subroutine names have an appended underscore.
It is enabled by default.

### Sparse Solvers

`EZP_ENABLE_MUMPS` and `EZP_ENABLE_LIS` control whether the corresponding sparse solvers are built.
Both are enabled by default.
`pardiso` is available whenever the `EZP_MKL` macro is defined, which happens automatically when MKL is found.

### OpenMP

`EZP_ENABLE_OPENMP` enables OpenMP for the sparse solvers.
It is enabled by default.
When enabled, `EZP_ENABLE_OPENMP_MUMPS`, `EZP_ENABLE_OPENMP_LIS`, and `EZP_ENABLE_OPENMP_GK` further control whether `MUMPS`, `Lis`, and `GKlib` link against OpenMP.

### Standalone Solvers

Use `-DEZP_STANDALONE=ON` in CMake to build the standalone solver executables.

### Testing

Use `-DEZP_TEST=ON` in CMake to build the test suite and the runnable examples.

## Sparse Solver Dependencies

The dense `ScaLAPACK`-based solvers (`pgesv`, `pposv`, `pgbsv`, etc.) are always available.
The sparse solvers require additional libraries and must be explicitly enabled.

### PARDISO

`pardiso` is bundled in Intel® oneMKL.
It is detected automatically when the oneAPI environment is sourced.
No additional CMake flag is required.
The `EZP_MKL` macro is defined when MKL is found.

### MUMPS

`MUMPS` is built from the bundled source in `external/mumps`.
It is controlled by `EZP_ENABLE_MUMPS` and defaults to on.

### Lis

`Lis` is built from the bundled source in `external/lis`.
It is controlled by `EZP_ENABLE_LIS` and defaults to on.
