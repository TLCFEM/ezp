# ezp

[![codecov](https://codecov.io/gh/TLCFEM/ezp/graph/badge.svg?token=ME0M312F5M)](https://codecov.io/gh/TLCFEM/ezp)
[![master](https://github.com/TLCFEM/ezp/actions/workflows/master.yml/badge.svg?branch=master)](https://github.com/TLCFEM/ezp/actions/workflows/master.yml)

`ezp` is a lightweight C++ wrapper for selected distributed solvers for linear systems.

## Features

1. easy to use interface
2. drop-in header-only library
3. standalone solver binaries that can be invoked by various callers
4. random tested implementation

The following solvers are implemented.

| availability | type of matrix                             | operation | solver  | package   |
|:------------:|--------------------------------------------|-----------|---------|-----------|
|      🗹      | general (partial pivoting)                 | simple    | PxGESV  | ScaLAPACK |
|      🗹      | general (partial pivoting)                 | expert    | PxGESVX | ScaLAPACK |
|      🗹      | symmetric/Hermitian positive definite      | simple    | PxPOSV  | ScaLAPACK |
|      🗹      | symmetric/Hermitian positive definite      | expert    | PxPOSVX | ScaLAPACK |
|      🗹      | general band (partial pivoting)            | simple    | PxGBSV  | ScaLAPACK |
|      🗹      | general band (no pivoting)                 | simple    | PxDBSV  | ScaLAPACK |
|      🗹      | symmetric/Hermitian positive definite band | simple    | PxPBSV  | ScaLAPACK |
|      🗹      | sparse                                     |           | PARDISO | MKL       |
|      🗹      | sparse                                     |           | MUMPS   | MUMPS     |

## Dependency

The `ezp` library requires C++ 20 compatible compiler.
The following drivers are needed.

1. an implementation of `LAPACK` and `BLAS`, such as `OpenBLAS`, `MKL`, etc.
2. an implementation of `ScaLAPACK`
3. an implementation of `MPI`, such as `OpenMPI`, `MPICH`, etc.

## Example

It is assumed that the root node (rank 0) prepares the left hand side $$A$$ and right hand side $$B$$.
The solvers distribute the matrices to available processes and solve the system, return the solution back to the master
node.

The solvers are designed in such a way that all `BLACS` and `ScaLAPACK` details are hidden.
One shall prepare the matrices (**on the root node**) and call the solver.
The following is a typical example.
It highly resembles the sequential version of how one would typically solve a linear system.

The following is a working example.

```cpp
#include <ezp/pgesv.hpp>
#include <iomanip>
#include <iostream>

using namespace ezp;

int main() {
    // get the current blacs environment
    const auto rank = get_env<int>().rank();

    constexpr auto N = 6, NRHS = 2;

    // storage for the matrices A and B
    std::vector<double> A, B;

    if(0 == rank) {
        // the matrices are only initialized on the root process
        A.resize(N * N, 0.);
        B.resize(N * NRHS, 1.);

        // helper functor to convert 2D indices to 1D indices
        // it's likely the matrices are provided by some other subsystem
        const auto IDX = par_dgesv<int>::indexer{N};

        for(auto I = 0; I < N; ++I) A[IDX(I, I)] = static_cast<double>(I);
    }

    // create a parallel solver
    // it takes the number of rows and columns of the process grid as arguments
    // or let the library automatically determine as follows
    // need to wrap the data in full_mat objects
    // it requires the number of rows and columns of the matrix, and a pointer to the data
    // on non-root processes, the data pointer is nullptr as the vector is empty
    // par_dgesv<int>().solve(full_mat{N, N, A.data()}, full_mat{N, NRHS, B.data()});
    par_dgesv<int>().solve({N, N, A.data()}, {N, NRHS, B.data()});

    if(0 == rank) {
        std::cout << std::setprecision(10) << "Solution:\n";
        for(auto i = 0; i < B.size(); ++i) std::cout << B[i] << '\n';
    }

    return 0;
}
```
