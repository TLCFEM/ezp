# Compilation

As `ezp` is wrapper library of `ScaLAPACK`, the following dependencies are necessary to compile any executables that use `ezp`.

1. an implementation of `LAPACK` and `BLAS`, such as `OpenBLAS`, `Intel® oneAPI Math Kernel Library (oneMKL)`, `AMD Optimizing CPU Libraries (AOCL)`, etc.
2. an implementation of `ScaLAPACK`, such as the reference implementation, Intel's implementation, NVIDIA's implementation, etc.
3. an implementation of `MPI`, such as `OpenMPI`, `MPICH`, `Intel® MPI` etc.

Before compiling the executables, one must ensure those libraries are available.